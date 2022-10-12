import torch
from torch import Tensor
from hcat.lib.utils import _crop
import torch.nn as nn

from typing import Tuple, Dict, Optional, List
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import triton
import triton.language as tl

import matplotlib.pyplot as plt

@triton.jit
def _embedding_forward_kernel(
        start_ptr,  # *Pointer* to first input vector
        output_ptr,  # *Pointer* to output vector
        X, Y, Z,
        b_stride,
        c_stride,
        x_stride,
        y_stride,  # Matrix Strides
        z_stride,
        x_scale,
        y_scale,
        z_scale,
        n_elements,  # Size of the vector
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(start_ptr + offsets, mask=mask)  # Load values of input tensor in memory
    vectors = tl.zeros(x.shape, dtype=tl.float16)  # Create a tensor for the offsets we want to calculate

    scale = tl.zeros(x.shape, dtype=tl.float16)  # Create a tensor for the offsets we want to calculate

    b_ind, c_ind, x_ind, y_ind, z_ind = get_index(offsets, b_stride, c_stride, x_stride, y_stride, z_stride)

    scale = tl.where(c_ind == 0, x_scale, scale)
    scale = tl.where(c_ind == 1, y_scale, scale)
    scale = tl.where(c_ind == 2, z_scale, scale)

    vectors = tl.where(c_ind == 0, x_ind, vectors)
    vectors = tl.where(c_ind == 1, y_ind, vectors)
    vectors = tl.where(c_ind == 2, z_ind, vectors)

    x = x * scale

    tl.store(output_ptr + offsets, x + vectors, mask=mask)

@triton.jit
def get_index(offsets, b_stride, c_stride, x_stride, y_stride, z_stride):

    b_ind = offsets // b_stride
    _offsets = offsets - (b_ind * b_stride)

    # We first account for batching!
    c_ind = offsets // c_stride  # Which channel are we in?
    _offsets = offsets - (c_ind * c_stride)

    # Write the X Index
    x_ind = _offsets // x_stride
    _offsets = _offsets - (x_ind * x_stride)

    # Write the Y Index
    y_ind = _offsets // y_stride
    _offsets = _offsets - (y_ind * y_stride)

    # Write the Z Index
    z_ind = _offsets // z_stride

    return b_ind, c_ind, x_ind, y_ind, z_ind

class Vec2Embed3D(Function):
    """
    Performs the vector to Embedding on 4D Inputs!
    """

    @staticmethod
    def forward(ctx, vec: torch.Tensor, scale: Tuple[float]):
        assert vec.ndim == 5
        assert vec.shape[1] == 3
        assert len(scale) == 3

        # We need to preallocate the output
        output = torch.empty_like(vec)

        assert vec.is_cuda and output.is_cuda
        assert vec.is_contiguous and output.is_contiguous

        n_elements = output.numel()

        grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)

        B, C, X, Y, Z = vec.shape
        _bs, _cs, _xs, _ys, _zs = vec.stride()
        _x, _y, _z = scale

        _embedding_forward_kernel[grid](vec, output,
                                        X=X, Y=Y, Z=Z,
                                        b_stride=_bs, c_stride=_cs, x_stride=_xs, y_stride=_ys, z_stride=_zs,
                                        x_scale=_x, y_scale=_y, z_scale=_z,
                                        n_elements=n_elements, BLOCK_SIZE=1024)


        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):

        scale = torch.tensor(ctx.scale, device=grad_outputs.device).view(1, 3, 1, 1, 1)
        grad_outputs = grad_outputs.mul(scale)

        return grad_outputs, None

fused_vector_to_embedding = Vec2Embed3D.apply

class VectorToEmbedding:
    def __init__(self, scale: Tensor = torch.tensor([50]), device='cpu'):

        self._device = device

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor([scale])
        self.scale = scale.to(self._device) if scale.numel() != 1 else torch.tensor([scale, scale, scale],
                                                                                    device=self._device).float()
        self.scale_tuple = (float(self.scale[0].item()),
                            float(self.scale[1].item()),
                            float(self.scale[2].item()))

        super(VectorToEmbedding, self).__init__()

    def __call__(self, vector: Tensor, n: int = 1) -> Tensor:

        # assert not torch.any(torch.isnan(vector))
        #
        # if vector.is_cuda and n == 1:
        #     """
        #     For whatever reason, it doenst handle batching. I dont know why...
        #     """
        #     out = [self._forward_fused(vector[[i], ...]) for i in range(vector.shape[0])]
        #
        #     torch.cuda.current_stream().synchronize()
        #
        #     for o in out:
        #         assert not torch.any(torch.isnan(o))
        #
        #     return torch.concat(out, dim=0)
        # else:
        return self._forward_torch(vector, n)

    @torch.jit.unused
    def _forward_fused(self, vector: Tensor):
        out = Vec2Embed3D.apply(vector, self.scale_tuple)
        return out

    def _forward_torch(self, vector: Tensor, n):
        """
        :param vector: [B, 3=[x,y,z], X, Y, Z] prediction vector from spatial embedding model
        :return: [B, 3=[x,y,z], X, Y, Z] projected spatial embedding vector
        """
        if vector.ndim != 5: raise RuntimeError('Expected input tensor ndim == 5')

        num: Tensor = torch.clone(self.scale.float())

        with torch.no_grad():
            _x = torch.linspace(0, vector.shape[2] - 1, vector.shape[2], device=vector.device)
            _y = torch.linspace(0, vector.shape[3] - 1, vector.shape[3], device=vector.device)
            _z = torch.linspace(0, vector.shape[4] - 1, vector.shape[4], device=vector.device)

            xv, yv, zv = torch.meshgrid([_x, _y, _z])

            mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                              yv.unsqueeze(0).unsqueeze(0),
                              zv.unsqueeze(0).unsqueeze(0)), dim=1)

        # Apply the mesh grid to the vector! We always do this at least once!
        vector = vector.mul(num.view(1, 3, 1, 1, 1))
        mesh = mesh + vector

        for _ in range(n - 1):  # Only executes if n > 1
            index = mesh.round()
            b, c, x, y, z = index.shape
            for i, k in enumerate([x, y, z]):
                index[:, i, ...] = torch.clamp(index[:, i, ...], 0, k)

            index = (index[:, [0], ...] * y * z) + (index[:, [1], ...] * z) + (index[:, [2], ...])
            out_of_bounds = torch.logical_and(index > x * y * z - 1, index < 0)
            index[out_of_bounds] = 0

            index = index.clamp(0, x * y * z - 1).long()
            # mesh = mesh + vector.flatten(start_dim=2)[]
            for i, shape in enumerate((x, y, z)):
                mesh[:, [i], ...] = mesh[:, [i], ...] + vector[:, [i], ...].take(index)
                out_of_bounds = torch.logical_or(mesh[:, [i], ...] > shape, mesh < 0)
                mesh[out_of_bounds] = -1

        return mesh

    def cuda(self):
        self._device = 'cuda:0'
        self.scale.to(self._device)

    def cpu(self):
        self._device = 'cpu'
        self.scale.to(self._device)

    def to(self, device: str):
        self._device = device
        self.scale.to(self._device)

    def __repr__(self):
        return f'nn.Module[name=VectorToEmbedding, scale={self.scale}]'


def _vec2emb(scale: Tensor, vector: Tensor, n: int) -> Tensor:
    if vector.ndim != 5: raise RuntimeError('Expected input tensor ndim == 5')

    num: Tensor = torch.clone(scale.float())

    with torch.no_grad():
        _x = torch.linspace(0, vector.shape[2] - 1, vector.shape[2], device=vector.device)
        _y = torch.linspace(0, vector.shape[3] - 1, vector.shape[3], device=vector.device)
        _z = torch.linspace(0, vector.shape[4] - 1, vector.shape[4], device=vector.device)

        xv, yv, zv = torch.meshgrid([_x, _y, _z])

        mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                          yv.unsqueeze(0).unsqueeze(0),
                          zv.unsqueeze(0).unsqueeze(0)), dim=1)

    # Apply the mesh grid to the vector! We always do this at least once!
    vector = vector.mul(num.view(1, 3, 1, 1, 1))
    mesh = mesh + vector

    # for _ in range(n - 1):  # Only executes if n > 1
    #     # pass
    #     index = torch.clone(mesh.round())
    #     b, c, x, y, z = index.shape
    #
    #     index[:, 0, ...] = torch.clamp(index[:, 0, ...], 0, x)
    #     index[:, 1, ...] = torch.clamp(index[:, 1, ...], 0, y)
    #     index[:, 2, ...] = torch.clamp(index[:, 2, ...], 0, z)

        # for i, k in enumerate([x, y, z]):
        #     index[:, i, ...] = torch.clamp(index[:, i, ...], 0, k)
        #
        # index = (index[:, [0], ...] * y * z) + (index[:, [1], ...] * z) + (index[:, [2], ...])
        # out_of_bounds = torch.logical_and(index > x * y * z - 1, index < 0)
        # index[out_of_bounds] = 0
        #
        # index = index.clamp(0, x * y * z - 1).long()
        # mesh = mesh + vector.flatten(start_dim=2)[]
        # for i, shape in enumerate((x, y, z)):
        #     mesh[:, [i], ...] = mesh[:, [i], ...] + vector[:, [i], ...].take(index)
        #     out_of_bounds = torch.logical_or(mesh[:, [i], ...] > shape, mesh < 0)
        #     mesh[out_of_bounds] = -1

    return mesh