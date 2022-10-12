from typing import Tuple, Dict, Optional, List
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple, Union

import triton
import triton.language as tl


@triton.jit
def _embedding_forward_kernel(
        output_ptr,  # [N, X, Y, Z]
        embed_ptr,  # *Pointer* to first input vector # [3, X, Y, Z]
        centroid_ptr,  # [N ,3] Use this to figure out which centroid?

        n_stride, x_stride, y_stride, z_stride,

        # Centroid Strides
        n_centroid_stride, coord_stride,

        # Sigma
        sigma_ptr,

        # Size of the vector
        embed_numel,
        output_numel,
        centroid_numel,

        # Constants
        BLOCK_SIZE: tl.constexpr
):
    """
    Effectivly Does This...

    _embed_grad = torch.zeros(ctx.embed.shape, dtype=torch.float32, device=grad_outputs.device)
    sigma = torch.tensor(ctx.sigma, device=grad_outputs.device)
    for n, center in enumerate(ctx.centroids):
        _embed_grad += 2 * (ctx.embed - torch.tensor(center, device=grad_outputs.device).view(3, 1, 1, 1)) / sigma.view(3, 1,1,1) * grad_outputs[[n], ...]
    return _embed_grad, None, None


    """
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    coord_channel = tl.program_id(axis=1)  # We use a 1D launch grid so axis is 0
    n_centroid = tl.program_id(axis=2)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) + (coord_channel * n_stride)

    n_ind, x_ind, y_ind, z_ind = get_index(offsets, n_stride, x_stride, y_stride, z_stride)

    # Create a mask to guard memory operations against out-of-bounds accesses
    embed_offsets = (coord_channel * n_stride) + (x_ind * x_stride) + (y_ind * y_stride) + (z_ind * z_stride)
    embed_mask = embed_offsets < embed_numel
    embed = tl.load(embed_ptr + embed_offsets, mask=embed_mask)

    centroid_offsets = (n_centroid * n_centroid_stride) + (coord_stride * coord_channel)
    centroid_mask = centroid_offsets < centroid_numel
    center = tl.load(centroid_ptr + centroid_offsets, mask=centroid_mask)

    sigma = tl.load(sigma_ptr + coord_channel)

    out = ((embed - center) * (embed - center)) / (sigma + 1e-16)

    output_offsets = (n_centroid * n_stride) + (x_ind * x_stride) + (y_ind * y_stride) + (z_ind * z_stride)
    output_mask = output_offsets < output_numel

    tl.atomic_add(output_ptr + output_offsets, out, mask=output_mask)


@triton.jit
def _embedding_backward_kernel(previous_grad_ptr, centroid_ptr, embed_ptr, grad_ptr,

                               n_stride, x_stride, y_stride, z_stride, n_centroid_stride, coord_stride,

                               sigma_ptr,

                               # Size of the vector
                               embed_numel, previous_grad_numel, centroid_numel,
                               # Constants
                               BLOCK_SIZE: tl.constexpr
                               ):
    """
    Effectivly Does This...

    _embed_grad = torch.zeros(ctx.embed.shape, dtype=torch.float32, device=grad_outputs.device)
    sigma = torch.tensor(ctx.sigma, device=grad_outputs.device)
    for n, center in enumerate(ctx.centroids):
        _embed_grad += 2 * (ctx.embed - torch.tensor(center, device=grad_outputs.device).view(3, 1, 1, 1)) / sigma.view(3, 1,1,1) * grad_outputs[[n], ...]
    return _embed_grad, None, None


    """
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    coord_channel = tl.program_id(axis=1)  # We use a 1D launch grid so axis is 0

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    n_ind, x_ind, y_ind, z_ind = get_index(offsets, n_stride, x_stride, y_stride, z_stride)

    # Create a mask to guard memory operations against out-of-bounds accesses
    previous_grad_mask = offsets < previous_grad_numel
    previous_grad = tl.load(previous_grad_ptr + offsets,
                            mask=previous_grad_mask)  # Load values of input tensor in memory

    embed_offsets = (coord_channel * n_stride) + (x_ind * x_stride) + (y_ind * y_stride) + (z_ind * z_stride)
    embed_mask = embed_offsets < embed_numel
    embed = tl.load(embed_ptr + embed_offsets, mask=embed_mask)

    centroid_offsets = (n_ind * n_centroid_stride) + (coord_stride * coord_channel)
    centroid_mask = centroid_offsets < centroid_numel
    center = tl.load(centroid_ptr + centroid_offsets, mask=centroid_mask)

    sigma = tl.load(sigma_ptr + coord_channel)

    grad = 2 * (embed - center) / sigma * previous_grad
    tl.atomic_add(grad_ptr + embed_offsets, grad, mask=embed_mask)


@triton.jit
def get_index(offsets, c_stride, x_stride, y_stride, z_stride):
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

    return c_ind, x_ind, y_ind, z_ind


class embed2prob3D(Function):
    """
    Performs the vector to Embedding on 4D Inputs!
    """

    @staticmethod
    def forward(ctx, embed: torch.Tensor, centroids: Tensor, sigma: Tensor):
        assert embed.ndim == 4
        assert embed.shape[0] == 3

        C, X, Y, Z = embed.shape
        _cs, _xs, _ys, _zs = embed.stride()

        output = torch.zeros((centroids.shape[0], X, Y, Z), device=embed.device, dtype=embed.dtype)

        assert embed.is_cuda and output.is_cuda
        assert embed.is_contiguous and output.is_contiguous

        centroid_stride = centroids.stride()

        sigma = (sigma ** 2) * -2

        assert sigma.max() < 0

        grid = lambda META: (triton.cdiv(X * Y * Z, META['BLOCK_SIZE']), 3, centroids.shape[0])

        _embedding_forward_kernel[grid](
            centroid_ptr=centroids,  # [N ,3] Use this to figure out which centroid?
            embed_ptr=embed,  # *Pointer* to first input vector # [3, X, Y, Z]
            output_ptr=output,  # *Pointer* to output vector # [3, X, Y, Z]

            # Vector Strides,
            n_stride=_cs, x_stride=_xs, y_stride=_ys, z_stride=_zs,

            # Centroid Strides
            n_centroid_stride=centroid_stride[0], coord_stride=centroid_stride[1],

            # Sigma
            sigma_ptr=sigma,

            # Size of the vector
            output_numel=output.numel(),
            embed_numel=embed.numel(),
            centroid_numel=centroids.numel(),

            # Constants
            BLOCK_SIZE=512)

        ctx.centroids = centroids
        ctx.embed = embed
        ctx.sigma = sigma

        return output

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        """
        SUM_{n centroids} = 2 * (vec - x) * sigma * grad_outputs


        # Native pyTorch implementation...
        _embed_grad = torch.zeros(ctx.embed.shape, dtype=torch.float32, device=grad_outputs.device)
        sigma = torch.tensor(ctx.sigma, device=grad_outputs.device)
        for n, center in enumerate(ctx.centroids):
            _embed_grad += 2 * (ctx.embed - torch.tensor(center, device=grad_outputs.device).view(3, 1, 1, 1)) / sigma.view(3, 1,1,1) * grad_outputs[[n], ...]
        return _embed_grad, None, None

        :param ctx:
        :param grad_outputs:
        :return:
        """

        assert grad_outputs.ndim == 4
        assert ctx.embed.ndim == 4
        assert ctx.embed.shape[0] == 3
        assert len(ctx.centroids) == grad_outputs.shape[0]
        assert ctx.embed.shape[1] == grad_outputs.shape[1]
        assert ctx.embed.shape[2] == grad_outputs.shape[2]
        assert ctx.embed.shape[3] == grad_outputs.shape[3]
        assert len(ctx.sigma) == 3

        C, X, Y, Z = ctx.embed.shape
        _cs, _xs, _ys, _zs = ctx.embed.stride()

        output = torch.zeros((3, X, Y, Z), device=grad_outputs.device, dtype=grad_outputs.dtype)

        _cos, _xos, _yos, _zos = output.stride()

        previous_grad_numel = grad_outputs.numel()

        centroids = ctx.centroids
        n_centroid_stride, coord_stride = centroids.stride()

        grid = lambda META: (triton.cdiv(previous_grad_numel, META['BLOCK_SIZE']), 3)  # 2D Lauch Grid!!!

        _embedding_backward_kernel[grid](
            previous_grad_ptr=grad_outputs,  # [N ,X, Y, Z]  # Iterating over this vector because its the biggest!
            centroid_ptr=centroids,  # [N ,3] Use this to figure out which centroid?
            embed_ptr=ctx.embed,  # *Pointer* to first input vector # [3, X, Y, Z]
            grad_ptr=output,  # *Pointer* to output vector # [3, X, Y, Z]

            # Vector Strides,
            n_stride=_cs, x_stride=_xs, y_stride=_ys, z_stride=_zs,

            # Centroid Strides
            n_centroid_stride=n_centroid_stride, coord_stride=coord_stride,

            # Sigma
            sigma_ptr=ctx.sigma,

            # Size of the vector
            embed_numel=ctx.embed.numel(),
            previous_grad_numel=grad_outputs.numel(),
            centroid_numel=centroids.numel(),

            # Constants
            BLOCK_SIZE=1024)

        return output, None, None


embed2prob = embed2prob3D.apply


class EmbeddingToProbability(nn.Module):
    def __init__(self):
        super(EmbeddingToProbability, self).__init__()

    def forward(self, embedding: Tensor,
                objective: Union[List[Tensor], Tensor, List[Dict[int, Tensor]]],
                sigma: Tensor,
                baked: Optional[bool] = False):

        objective = [objective] if not isinstance(objective, List) else objective # might be centroids or skeletons

        if isinstance(objective[0], Tensor) and not baked:
            return self._forward_torch(embedding, objective, sigma)

        if isinstance(objective[0], Tensor) and baked:
            return self._forward_baked(embedding, objective, sigma)

        elif isinstance(objective[0], dict):
            return self._forward_skeletons(embedding, objective, sigma)
        else:
            raise RuntimeError(f'Expected dict or list of Tensor not {type(objective[0])=}')

    @torch.jit.ignore
    def _forward_triton(self, embedding, centroids, sigma):
        return torch.exp(embed2prob(embedding, centroids, sigma))

    def _forward_torch(self, embedding: Tensor, centroids: List[Tensor], sigma: Tensor) -> List[Tensor]:
        """
        Calculates the euclidean distance between the centroid and the embedding
        embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
        euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

                             /    (e_ix - C_kx)^2       (e_iy - C_ky)^2        (e_iz - C_kz)^2   \
          prob_k(e_i) = exp |-1 * ----------------  -  -----------------   -  ------------------  |
                            \     2*sigma_kx ^2         2*sigma_ky ^2          2 * sigma_kz ^2  /

        Example:

        >>> from hcat.lib.functional import EmbeddingToProbability
        >>> import torch
        >>> embed = torch.load('embed.trch') # [B, C, X, Y, Z]
        >>> emb2prob = torch.jit.script(EmbeddingToProbability)
        >>> probability = emb2prob(embed, centroids, sigma)


        :param embedding: [B, K=3, X, Y, Z] embedding tensor where K is the likely centroid component: {X, Y, Z}
        :param centroids: List[[N, K_true=3]] object centroids where N is the number of instances in the image and K_true is centroid {x, y, z}
        :param sigma: Tensor of shape = (1) or (embedding.shape)
        :return: [B, N, X, Y, Z] of probabilities for instance N
        """

        sigma = sigma + torch.tensor([1e-16], device=embedding.device)  # when sigma goes to zero, things tend to break

        if sigma.numel() == 1:
            sigma = torch.cat((sigma, sigma, sigma), dim=0)

        # Sometimes centroids might be in pixel coords instead of scaled.
        # If so, scale by num (usually 512)
        # centroids: List[Tensor] = [c / num if c.max().gt(5) else c for c in centroids]

        assert embedding.shape[0] == len(centroids), embedding.shape

        b, _, x, y, z = embedding.shape
        batch_list = torch.jit.annotate(List[Tensor], [])
        newshape = (1, 3, 1, 1, 1)

        # Common operation. Done outside of loop for speed.
        sigma = sigma.pow(2).mul(2).view(newshape).mul(-1)

        for batch_index in range(b):
            n, _ = centroids[batch_index].shape
            prob_list = torch.jit.annotate(List[Tensor], [])

            # Calculate euclidean distance between centroid and embedding for each pixel and
            # turn that distance to probability and put it in preallocated matrix for each n
            # In eval mode uses in place operations to save memory!
            # A bit scuffed but should optimize well in torchscript
            for i in range(n):
                prob_list += [
                    torch.exp((embedding[batch_index, ...] - centroids[batch_index][i, ...].view((1, 3, 1, 1, 1)))
                              .pow(2)
                              .div(sigma)
                              .sum(dim=1)).squeeze(1)]

            batch_list.append(torch.stack(prob_list, dim=1))

        return batch_list

    def _forward_baked(self, embedding: Tensor, baked: List[Tensor], sigma: Tensor) -> List[Tensor]:
        """
        Calculates the euclidean distance between the centroid and the embedding
        embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
        euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

                             /    (e_ix - C_kx)^2       (e_iy - C_ky)^2        (e_iz - C_kz)^2   \
          prob_k(e_i) = exp |-1 * ----------------  -  -----------------   -  ------------------  |
                            \     2*sigma_kx ^2         2*sigma_ky ^2          2 * sigma_kz ^2  /

        Example:

        >>> from hcat.lib.functional import EmbeddingToProbability
        >>> import torch
        >>> embed = torch.load('embed.trch') # [B, C, X, Y, Z]
        >>> emb2prob = torch.jit.script(EmbeddingToProbability)
        >>> probability = emb2prob(embed, centroids, sigma)


        :param embedding: [B, K=3, X, Y, Z] embedding tensor where K is the likely centroid component: {X, Y, Z}
        :param centroids: List[[N, K_true=3]] object centroids where N is the number of instances in the image and K_true is centroid {x, y, z}
        :param sigma: Tensor of shape = (1) or (embedding.shape)
        :return: [B, N, X, Y, Z] of probabilities for instance N
        """
        baked: Tensor = baked[0] if isinstance(baked, list) else baked

        sigma = sigma + torch.tensor([1e-16], device=embedding.device)  # when sigma goes to zero, things tend to break

        if sigma.numel() == 1:
            sigma = torch.cat((sigma, sigma, sigma), dim=0)

        b, _, x, y, z = embedding.shape
        newshape = (1, 3, 1, 1, 1)

        # Common operation. Done outside of loop for speed.
        sigma = sigma.pow(2).mul(2).view(newshape).mul(-1)

        out = torch.exp((embedding - baked).pow(2).div(sigma).sum(dim=1, keepdim=True))

        return out

    @torch.jit.ignore
    def _forward_skeletons(self, embedding: Tensor, skeleton: List[Dict[int, Tensor]], sigma: Tensor) -> List[Tensor]:
        """
        Calculates the euclidean distance between the skeleton and the embedding
        embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
        euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

                             /    (e_ix - C_kx)^2       (e_iy - C_ky)^2        (e_iz - C_kz)^2   \
          prob_k(e_i) = exp |-1 * ----------------  -  -----------------   -  ------------------  |
                            \     2*sigma_kx ^2         2*sigma_ky ^2          2 * sigma_kz ^2  /




        :param embedding: [B, K=3, X, Y, Z] embedding tensor where K is the likely centroid component: {X, Y, Z}
        :param skeleton List[Dict[int, Tensor]] Batch list of Skeletons of shape [N, 3={xyz}]
        :param sigma: Tensor of shape = (1) or (embedding.shape)
        :return: [B, N, X, Y, Z] of probabilities for instance N
        """

        sigma = sigma + torch.tensor([1e-16], device=embedding.device)  # when sigma goes to zero, things tend to break

        if sigma.numel() == 1:
            sigma = torch.cat((sigma, sigma, sigma), dim=0)

        b, _, x, y, z = embedding.shape
        batch_list = torch.jit.annotate(List[Tensor], [])
        newshape = (1, 3, 1, 1, 1)

        # Common operation. Done outside of loop for speed.
        sigma = sigma.pow(2).mul(2).view(newshape).mul(-1)

        # We iterate over the entire batch list...
        for batch_index in range(b):
            prob_list = []
            # For each instance we have a skeleton of multiple points. We dont vectorize to save space...
            for i, key in enumerate(skeleton[batch_index]):  # val
                b, c, x, y, z = embedding.shape
                point: Tensor = skeleton[batch_index][key]  # Shape [N, 3]

                assert point.shape[0] > 0, f'{point.shape=}, {key=}, {i=}'

                # Get the skeleton points which seem most reasonable...
                ind = torch.sum(torch.logical_or(point < -25, point > embedding.shape[2] + 5), dim=1).gt(0)
                point = point[torch.logical_not(ind), :]

                # Iterate over each point in the skeleton and create a prob map for that...
                prob = None
                for j in range(point.shape[0]):
                    _prob = self._gauss(embedding[batch_index, ...], point[j, :], sigma)
                    prob = _prob if j == 0 else torch.where(prob > _prob, prob, _prob)

                prob = prob if prob is not None else torch.zeros((1, x, y, z), device=embedding.device)
                prob_list += [prob]

            if len(prob_list) == 0:
                b, c, x, y, z = embedding.shape
                prob_list = [torch.zeros((1, x, y, z), device=embedding.device)]

            batch_list.append(torch.stack(prob_list, dim=1))

        return batch_list

    @staticmethod
    @torch.jit.script
    def _gauss(embed: Tensor, point: Tensor, sigma: Tensor) -> Tensor:
        prob = torch.exp((embed - point.view((1, 3, 1, 1, 1)))
                         .pow(2)
                         .div(sigma)
                         .sum(dim=1)).squeeze(1)
        return prob


if __name__ == '__main__':
    to_prob = EmbeddingToProbability()
    centroids = torch.tensor([[339.7057, 279.6797, 8.5821],
                              # [326.4857, 259.7429, 7.0000],
                              # [308.4445, 237.6667, 7.0000],
                              # [289.4868, 216.2895, 7.1316],
                              # [272.1857, 193.3214, 7.7500],
                              # [257.6667, 175.1667, 7.0000],
                              # [241.0000, 152.3333, 7.0000],
                              # [223.5454, 128.4000, 7.3273],
                              # [206.8846, 107.1538, 7.0000],
                              # [187.6286, 84.2857, 7.0000],
                              # [172.1667, 61.6667, 7.0000],
                              # [156.0625, 38.8125, 7.0000],
                              # [140.8333, 18.0000, 7.0000],
                              # [126.9796, -3.0000, 7.1837],
                              # [303.8831, 296.4026, 10.9610],
                              # [287.0000, 276.2000, 10.7500],
                              # [272.5874, 256.5533, 9.6695],
                              # [250.8000, 232.9333, 10.7333],
                              # [233.2046, 212.7727, 11.0000],
                              # [229.6102, 183.5593, 7.6102],
                              # [213.0000, 163.7273, 7.7273],
                              # [186.9601, 143.7075, 9.6987],
                              # [170.7963, 122.2470, 9.3693],
                              # [161.0000, 95.0909, 8.1818],
                              # [138.3950, 77.1906, 9.4214],
                              # [123.2297, 54.4713, 9.2040],
                              # [106.9206, 30.4111, 9.1540],
                              # [89.9091, 8.5454, 10.0000],
                              # [75.7273, -15.8182, 10.0000],
                              # [271.6923, 298.8462, 9.0000],
                              # [246.6977, 278.4496, 9.8140],
                              # [228.5119, 255.9881, 9.4524],
                              # [221.8630, 233.4384, 8.1096],
                              # [199.1509, 213.2453, 9.0943],
                              # [193.0392, 189.9150, 7.8366],
                              # [169.0191, 170.6555, 9.0431],
                              # [150.2396, 149.5781, 9.2500],
                              # [135.0024, 123.7735, 9.3205],
                              # [113.9188, 102.9705, 9.4244],
                              # [101.5341, 78.3433, 8.5114],
                              # [86.3342, 55.2914, 8.0053],
                              # [61.1749, 31.6547, 9.4664],
                              # [42.1952, 7.8355, 9.8839],
                              # [24.3041, -13.7629, 10.0000],
                              # [302.3333, 161.2000, 9.0667],
                              # [284.3108, 135.0676, 9.4324]], device='cuda')
                              ], device='cuda')

    sigma = torch.tensor([5, 5, 3], device='cuda')

    embed = torch.load('embed.trch').cuda()
    print(embed.max(), embed.min())

    # embed = torch.rand(embed.shape, device='cuda') * 500

    out = to_prob(embed.unsqueeze(0), [centroids], sigma)
    torch.cuda.current_stream().synchronize()

    assert not torch.any(torch.isnan(out[0]))
    assert not torch.any(torch.isinf(out[0]))
