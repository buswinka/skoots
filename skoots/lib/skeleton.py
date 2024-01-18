from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from skoots.lib.morphology import (
    gauss_filter,
    binary_dilation_2d,
    _compute_zero_padding,
    _get_binary_kernel3d,
)
from skoots.lib.utils import get_cached_ball_coords, get_cached_disk_coords
from torch import Tensor
import logging


@torch.jit.script
def average_baked_skeletons(baked_skeleton: Tensor, kernel_size: int = 3) -> Tensor:
    """
    Takes a baked skeleton computed by skoots.lib.skeleton.bake_skeleton and averages all the values such that
    there is a smooth transition from one location to another.

    Shapes:
        - baked_skeleton :math:`(B_{in}, 3, X_{in}, Y_{in}, Z_{in})`
        - returns :math:`(B_{in}, 3, X_{in}, Y_{in}, Z_{in})`

    :param baked_skeleton: Baked skeleton tensor
    :param kernel_size: Kernel size for smoothing
    :return: smoothed skeleton: Smoothed Tensor
    """
    padding: Tuple[int, int, int] = _compute_zero_padding(
        (kernel_size, kernel_size, kernel_size)
    )
    kernel: Tensor = _get_binary_kernel3d(kernel_size, str(baked_skeleton.device))
    b, c, h, w, d = baked_skeleton.shape

    # map the local window to single vector
    features: Tensor = F.conv3d(
        baked_skeleton.reshape(b * c, 1, h, w, d), kernel, padding=padding, stride=1
    )
    features: Tensor = features.view(b, c, -1, h, w, d)  # B, C, -1, X, Y, Z

    nonzero = features.gt(0).sum(2)  # B, C, X, Y, Z
    nonzero[nonzero.eq(0)] = 1.0
    features = features.sum(
        2
    )  # This is the average of the kernel window... without zeros
    features = features / nonzero

    return features


@triton.jit
def _min_skeleton_kernel(
    # Pointers to mat
    mask_pointer,
    skeleton_pointer,
    skeleton_id_map_pointer,
    skeleton_len_pointer,
    out_pointer,
    anisotopy_pointer,  # 1D tensor
    # shapes of the MASK/OUT
    mask_x_shape,
    mask_y_shape,
    mask_z_shape,
    out_c_shape,  # should be 3
    # strides of the MASK/OUT
    anisotropy_stride,
    mask_x_stride,
    mask_y_stride,
    mask_z_stride,
    out_c_stride,  # should be the x,y,z channel of out
    out_x_stride,
    out_y_stride,
    out_z_stride,
    # skeleton shape
    n_skeleton: tl.constexpr,
    dim_skeleton,  # should always be three
    # skeleton strides
    # id_skeleton_stride, # skeleton will be a [N_objects, M_points, C=3] Tensor
    n_skeleton_stride,
    m_skeleton_stride,
    dim_skeleton_stride,

    SKEL_BLOCK_SIZE: tl.constexpr,
    ID_BLOCK_SIZE: tl.constexpr,
):
    """
    This is a triton kernel for calculating the closest skeleton vertex of a to a voxel in an instance mask, where each
    ID has a unique skeleton. Each instance mask voxel is presumed to contain a 0 (background) or integer id,
    signifying this voxel belongs to the object with that id.

    The mask is a 3D tensor of shape (X, Y, Z) with N objects in it. Each N object has a unique ID, but is not
    required to be sequential. I.e. there can be two IDs in mask: 234 and 6.

    Each objects skeleton consists of 3D points; with the number of points varrying between objects.
    This kernel combines

    :param mask_pointer:
    :param skeleton_pointer:
    :param skeleton_id_map_pointer:
    :param skeleton_len_pointer:
    :param out_pointer:
    :param anisotopy_pointer:
    :param mask_x_shape:
    :param mask_y_shape:
    :param mask_z_shape:
    :param out_c_shape:
    :param anisotropy_stride:
    :param mask_x_stride:
    :param mask_y_stride:
    :param mask_z_stride:
    :param out_c_stride:
    :param out_x_stride:
    :param out_y_stride:
    :param out_z_stride:
    :param n_skeleton:
    :param dim_skeleton:
    :param n_skeleton_stride:
    :param m_skeleton_stride:
    :param dim_skeleton_stride:
    :param SKEL_BLOCK_SIZE:
    :param ID_BLOCK_SIZE:
    :return:
    """

    # padding

    x0 = tl.program_id(axis=0)
    y0 = tl.program_id(axis=1)
    z0 = tl.program_id(axis=2)

    mask_center = tl.load(
        mask_pointer + (x0 * mask_x_stride + y0 * mask_y_stride + z0 * mask_z_stride)
    )

    if (
        mask_center == 0.0
    ):  # depending on how triton does this, it might be scuffed
        return

    dx = tl.load(anisotopy_pointer + (0 * anisotropy_stride))  # this should always be 3
    dy = tl.load(anisotopy_pointer + (1 * anisotropy_stride))  # this should always be 3
    dz = tl.load(anisotopy_pointer + (2 * anisotropy_stride))  # this should always be 3

    _off = tl.arange(0, ID_BLOCK_SIZE)  # assume stride is one here. Verified elsewhere
    skeleton_id_map = tl.load(skeleton_id_map_pointer + _off, mask=_off < n_skeleton)
    skeleton_len_map = tl.load(skeleton_len_pointer + _off, mask=_off < n_skeleton)

    tl.device_assert(tl.sum(skeleton_len_map > 0) == n_skeleton, 'it is broken')


    index = tl.argmax(skeleton_id_map == mask_center, axis=0)

    tl.device_assert(mask_center != 0, 'mask center is zero!')
    tl.device_assert(tl.sum(skeleton_id_map == mask_center) > 0, 'not finding skeleton id')


    dummy = tl.zeros_like(skeleton_len_map)

    tl.device_assert(tl.sum(tl.where( skeleton_id_map==mask_center, skeleton_len_map, dummy)) > 0, 'where is busted')

    N_element_skel = tl.max(
        tl.where(skeleton_id_map == mask_center, skeleton_len_map, dummy)
    )

    tl.device_assert(N_element_skel > 0, 'skeleton n elements at id is zero')


    _off = tl.arange(0, SKEL_BLOCK_SIZE)

    skel_x_ptr = skeleton_pointer + _off * m_skeleton_stride + (0 * dim_skeleton_stride) + (index * n_skeleton_stride)
    skel_y_ptr = skeleton_pointer + _off * m_skeleton_stride + (1 * dim_skeleton_stride) + (index * n_skeleton_stride)
    skel_z_ptr = skeleton_pointer + _off * m_skeleton_stride + (2 * dim_skeleton_stride) + (index * n_skeleton_stride)

    # pid is the center px location.
    skl_x = tl.load(skel_x_ptr, _off < N_element_skel)
    skl_y = tl.load(skel_y_ptr, _off < N_element_skel)
    skl_z = tl.load(skel_z_ptr, _off < N_element_skel)

    dist = (
        (skl_x - x0) * (skl_x - x0) * dx
        + (skl_y - y0) * (skl_y - y0) * dy
        + (skl_z - z0) * (skl_z - z0) * dz
    )  # has shape of N_SKEL

    min_dist = tl.min(dist, axis=0)

    # out = tl.zeros((3, n_skeleton), dtype=tl.float16)
    _zeros = tl.zeros((SKEL_BLOCK_SIZE,), dtype=tl.float16)

    closest_x = tl.max(tl.where(dist == min_dist, skl_x, _zeros), axis=0)
    closest_y = tl.max(tl.where(dist == min_dist, skl_y, _zeros), axis=0)
    closest_z = tl.max(tl.where(dist == min_dist, skl_z, _zeros), axis=0)
    #
    # # store_x
    tl.store(
        out_pointer
        + (0 * out_c_stride)
        + (x0 * out_x_stride)
        + (y0 * out_y_stride)
        + (z0 * out_z_stride),
        closest_x,
    )

    # store_y
    tl.store(
        out_pointer
        + (1 * out_c_stride)
        + (x0 * out_x_stride)
        + (y0 * out_y_stride)
        + (z0 * out_z_stride),
        closest_y,
    )

    # store_z
    tl.store(
        out_pointer
        + (2 * out_c_stride)
        + (x0 * out_x_stride)
        + (y0 * out_y_stride)
        + (z0 * out_z_stride),
        closest_z,
    )


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _bake_skeleton_triton(
    masks: Tensor,
    skeletons: Dict[int, Tensor],
    anisotropy: List[float],
    average: bool = True,
):
    """
    Launches a triton kernel to perform an in place distance calculation

    :param masks:
    :param skeletons: [N, 3]
    :param anisotropy:
    :param average:
    :return:
    """

    assert (
        masks.ndim == 3
    ), f"masks must have have 3 dimensions, not shape: {masks.shape}"
    assert masks.is_cuda, "masks must be on cuda"
    assert masks.is_contiguous(), "mask must be contiguous"
    for k, v in skeletons.items():
        assert v.is_cuda, "all skeletons must be on cuda device"
        assert v.is_contiguous, "all skeletons must be contiguous"
    assert len(anisotropy) == 3, "anisotropy should have 3 values"

    with torch.cuda.device(masks.device):
        x, y, z = masks.shape

        masks = masks.contiguous()
        baked = (
            torch.zeros((3, x, y, z)).to(masks.device).to(torch.float16).contiguous()
        )

        anisotropy = torch.tensor(anisotropy, device=masks.device).contiguous()

        skeleton_len_tensor = torch.zeros(len(skeletons), device=masks.device) # the size of each individual skeleton
        skeleton_id_map_tensor = torch.zeros(len(skeletons), device=masks.device) # the mapping of unique id to index in skeleton

        max_shape = max(v.shape[0] for v in skeletons.values()) if skeletons else 0
        if max_shape == 0:
            return baked

        combined_skeleton_tensors = torch.zeros((len(skeletons), max_shape, 3), device=masks.device) # [N_skel, M_points, 3]

        # Iterate over to place data in propper place...
        for i, (k,v) in enumerate(skeletons.items()):
            skeleton_len_tensor[i] = v.shape[0]  # place len of skel
            skeleton_id_map_tensor[i] =  k          # what index at each len
            combined_skeleton_tensors[i, 0:v.shape[0], :] = v

        assert skeleton_len_tensor.stride(0) == 1
        assert skeleton_id_map_tensor.stride(0) == 1


        num_out = 3
        grid = (x, y, z)

        N_skeletons = len(skeletons)
        dim_skel=3
        _min_skeleton_kernel[grid](
            mask_pointer=masks,
            skeleton_pointer=combined_skeleton_tensors,
            skeleton_id_map_pointer=skeleton_id_map_tensor,
            skeleton_len_pointer=skeleton_len_tensor,
            out_pointer=baked,
            anisotopy_pointer=anisotropy,
            mask_x_shape=x,
            mask_y_shape=y,
            mask_z_shape=z,
            mask_x_stride=masks.stride(0),
            mask_y_stride=masks.stride(1),
            mask_z_stride=masks.stride(2),
            out_c_stride=baked.stride(0),
            out_x_stride=baked.stride(1),
            out_y_stride=baked.stride(2),
            out_z_stride=baked.stride(3),
            out_c_shape=num_out,
            n_skeleton_stride=combined_skeleton_tensors.stride(0),
            m_skeleton_stride=combined_skeleton_tensors.stride(1),
            dim_skeleton_stride=combined_skeleton_tensors.stride(2),
            anisotropy_stride=anisotropy.stride(0),
            n_skeleton=N_skeletons,
            dim_skeleton=dim_skel,
            SKEL_BLOCK_SIZE=next_power_of_2(int(max_shape)),
            ID_BLOCK_SIZE=next_power_of_2(int(N_skeletons)),
        )
        torch.cuda.synchronize(masks.device)

    return baked


@torch.jit.ignore
def _bake_skeleton_torch(
    masks: Tensor,
    skeletons: Dict[int, Tensor],
    anisotropy: List[float] = (1.0, 1.0, 1.0),
    average: bool = True,
    device: str = "cpu",
) -> Tensor:
    r"""
    For each pixel :math:`p_ik` of object :math:`k` at index :math:`i\in[x,y,z]` in masks, returns a baked skeleton
    where the value at each index is the closest skeleton point :math:`s_{jk}` of any instance :math:`k`.

    This should reflect the ACTUAL spatial distance of your dataset for best results...These models tend to like XY
    embedding vectors more than Z. For anisotropic datasets, you should roughly provide the anisotropic correction
    factor of each voxel. For instance anisotropy of (1.0, 1.0, 5.0) means that the Z dimension is 5x larger than XY.


    Formally, the value at each position :math:`i\in[x,y,z]` of the baked skeleton tensor :math:`S` is the minimum of the
    euclidean distance function :math:`f(a, b)` and the skeleton point of any instance:

    .. math::
        S_{i} = min \left( f(i, s_{k})\right)\ for\ k \in [1, 2, ..., N]

    Shapes:
        - masks: :math:`(1, X_{in}, Y_{in}, Z_{in})`
        - skeletons: :math:`(3, N_i)`
        - anisotropy: :math:`(3)`
        - returns: :math:`(3, X_{in}, Y_{in}, Z_{in})`

    :param masks: Ground Truth instance mask of shape [1, X, Y, Z] of objects where each pixel is an integer id value.
    :param skeletons: Dict of skeleton indicies where each key is a unique instance of an object in mask.
        - Each skeleton has a shape [3, N] where N is the number of pixels constituting the skeleton
    :param anisotropy: Anisotropic correction factor for min distance calculation
    :param average: Average the skeletons such that there is a smooth transition form one area to the next
    :param device: torch.Device by which to run calculations

    :return: Baked skeleton
    """

    baked = torch.zeros(
        (3, masks.shape[0], masks.shape[1], masks.shape[2]), device=device
    )

    assert baked.device == masks.device, "Masks device must equal kwarg device"

    unique = torch.unique(masks)

    anisotropy = torch.tensor(anisotropy, device=device).view(1, 1, 3)

    for id in unique[unique != 0]:
        nonzero: Tensor = masks.eq(id).nonzero().unsqueeze(0).float()  # 1, N, 3
        skel: Tensor = skeletons[int(id)].to(device).unsqueeze(0).float()  # 1, N, 3

        # Calculate the distance between the skeleton of object 'id'
        # and all nonzero pixels of the binary mask of instance 'id'
        # print(skel.shape, nonzero.shape, id)

        # THIS IS SLOW!
        ind: Tensor = (
            torch.cdist(x1=skel.mul(anisotropy), x2=nonzero.mul(anisotropy))
            .squeeze(0)
            .argmin(dim=0)
        )

        nonzero = nonzero.squeeze(0).long()  # Can only index with long dtype

        baked[:, nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]] = (
            skel[0, ind, :].float().T
        )
        del ind

    return baked


def bake_skeleton(
    masks: Tensor,
    skeletons: Dict[int, Tensor],
    anisotropy: List[float] = (1.0, 1.0, 1.0),
    average: bool = True,
    device: str = "cpu",
):
    r"""
    For each pixel :math:`p_ik` of object :math:`k` at index :math:`i\in[x,y,z]` in masks, returns a baked skeleton
    where the value at each index is the closest skeleton point :math:`s_{jk}` of any instance :math:`k`.

    This should reflect the ACTUAL spatial distance of your dataset for best results...These models tend to like XY
    embedding vectors more than Z. For anisotropic datasets, you should roughly provide the anisotropic correction
    factor of each voxel. For instance anisotropy of (1.0, 1.0, 5.0) means that the Z dimension is 5x larger than XY.


    Formally, the value at each position :math:`i\in[x,y,z]` of the baked skeleton tensor :math:`S` is the minimum of the
    euclidean distance function :math:`f(a, b)` and the skeleton point of any instance:

    .. math::
        S_{i} = min \left( f(i, s_{k})\right)\ for\ k \in [1, 2, ..., N]

    Shapes:
        - masks: :math:`(1, X_{in}, Y_{in}, Z_{in})`
        - skeletons: :math:`(3, N_i)`
        - anisotropy: :math:`(3)`
        - returns: :math:`(3, X_{in}, Y_{in}, Z_{in})`

    :param masks: Ground Truth instance mask of shape [1, X, Y, Z] of objects where each pixel is an integer id value.
    :param skeletons: Dict of skeleton indicies where each key is a unique instance of an object in mask.
        - Each skeleton has a shape [3, N] where N is the number of pixels constituting the skeleton
    :param anisotropy: Anisotropic correction factor for min distance calculation
    :param average: Average the skeletons such that there is a smooth transition form one area to the next
    :param device: torch.Device by which to run calculations

    :return: Baked skeleton
    """
    if -1 in skeletons:
        x, y, z = masks.shape
        baked = (
            torch.zeros((3, x, y, z)).to(masks.device).to(torch.float16).contiguous()
        )
        return baked


    if masks.is_cuda:
        logging.debug(f'skoots.lib.skeletons.bake_skeleton() | applying triton kernel for skeleton baking')
        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks.squeeze(0).contiguous()
        baked = _bake_skeleton_triton(masks, skeletons, anisotropy)
    else:
        logging.debug(f'skoots.lib.skeletons.bake_skeleton() | applying pure torch kernel for skeleton baking')
        baked = _bake_skeleton_torch(masks, skeletons, anisotropy, device)

    logging.debug(f'skoots.lib.skeletons.bake_skeleton() | averaging baked skeletons')
    baked = average_baked_skeletons(baked.unsqueeze(0).float()).squeeze(0) if average else baked
    # baked[baked == 0] = -100

    return baked


def skeleton_to_mask(
    skeletons: Dict[int, Tensor],
    shape: Tuple[int, int, int],
    device: torch.device | str = None
) -> Tensor:
    r"""
    Converts a skeleton Dict to a skeleton mask which can simply be regressed against via Dice loss or similar...

    Shapes:
        - skeletons: [N, 3]
        - shape :math:`(3)`
        - returns: math:`(1, X_{in}, Y_{in}, Z_{in})`

    :param skeletons: Dict of skeletons
    :param shape: Shape of final mask
    :param device: output device
    :return: Mask of embedded skeleton px
    """

    skeleton_mask = None
    cached_inds = None

    if -1 in skeletons:
        return torch.zeros(shape, device)

    for k, v in skeletons.items():
        if skeleton_mask is None:
            skeleton_mask = torch.zeros(shape, device=v.device)
            cached_inds: Tensor = get_cached_disk_coords(device=v.device)

        # cached_inds = cached_inds[:, torch.logical_and(cached_inds[2,:].gt(-1), cached_inds[2,:].lt(1))]

        skeleton_indicies = (v.T.unsqueeze(1) + cached_inds.unsqueeze(2)).reshape(3, -1).long()
        # skeleton_indicies = v
        #
        x: Tensor = skeleton_indicies[0, :].long()
        y: Tensor = skeleton_indicies[1, :].long()
        z: Tensor = skeleton_indicies[2, :].long()

        ind_x = torch.logical_and(x >= 0, x < shape[0])
        ind_y = torch.logical_and(y >= 0, y < shape[1])
        ind_z = torch.logical_and(z >= 0, z < shape[2])

        ind = (
            ind_x.float() + ind_y.float() + ind_z.float()
        ) == 3  # Equivalent to a 3 way logical and

        skeleton_mask[
            skeleton_indicies[0, ind], skeleton_indicies[1, ind], skeleton_indicies[2, ind]
        ] = 1.0

    if skeleton_mask is None:  # in case skeletons has nothing in it
        skeleton_mask = torch.zeros(shape)

    return skeleton_mask.unsqueeze(0)


def _skeleton_to_mask(
    skeletons: Dict[int, Tensor],
    shape: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int] = (15, 15, 1),
    n: int = 2,
) -> Tensor:
    r"""
    Converts a skeleton Dict to a skeleton mask which can simply be regressed against via Dice loss or whatever...

    Shapes:
        - skeletons: [N, 3]
        - shape :math:`(3)`
        - returns: math:`(1, X_{in}, Y_{in}, Z_{in})`

    :param skeletons: Dict of skeletons
    :param shape: Shape of final mask
    :return: Maks of embedded skeleton px
    """
    skeleton_mask = torch.zeros(shape)

    for k in skeletons:
        x = skeletons[k][:, 0].long()
        y = skeletons[k][:, 1].long()
        z = skeletons[k][:, 2].long()

        ind_x = torch.logical_and(x >= 0, x < shape[0])
        ind_y = torch.logical_and(y >= 0, y < shape[1])
        ind_z = torch.logical_and(z >= 0, z < shape[2])

        ind = (
            ind_x.float() + ind_y.float() + ind_z.float()
        ) == 3  # Equivalent to a 3 way logical and

        skeleton_mask[x[ind], y[ind], z[ind]] = 1

    skeleton_mask = skeleton_mask.unsqueeze(0).unsqueeze(0)

    for _ in range(
        n
    ):  # this might make things a bit better on the skeleton side of things...
        skeleton_mask = gauss_filter(
            binary_dilation_2d(skeleton_mask.gt(0.5).float()),
            kernel_size,
            [0.8, 0.8, 0.8],
        )

    return skeleton_mask.squeeze(0)


def index_skeleton_by_embed(skeleton: Tensor, embed: Tensor) -> Tensor:
    r"""
    Returns an instance mask by indexing skeleton with an embedding tensor
    For memory efficiency, skeleton is only ever Referenced! Never copied (I hope)

    Shapes:
        - skeleton: :math:`(B_{in}=1, 1, X_{in}, Y_{in}, Z_{in})`
        - embed: :math:`(B_{in}=1, 3, X_{in}, Y_{in}, Z_{in})`

    :param skeleton: Skeleton of a single instance
    :param embed: Embedding
    :return: torch.int instance mask
    """
    assert embed.device == skeleton.device, "embed and skeleton must be on same device"
    # assert embed.shape[2::] == skeleton.shape[2::], 'embed and skeleton must have identical spatial dimensions'
    assert (
        embed.ndim == 5 and skeleton.ndim == 5
    ), "Embed and skeleton must be a 5D tensor"

    b, c, x, y, z = embed.shape  # get the shape of the embedding
    embed = embed.view(
        (c, -1)
    ).round()  # flatten the embedding to extract it as an index

    # We need to only select indicies which lie within the skeleton
    x_ind = embed[0, :].clamp(0, skeleton.shape[2] - 1).long()
    y_ind = embed[1, :].clamp(0, skeleton.shape[3] - 1).long()
    z_ind = embed[2, :].clamp(0, skeleton.shape[4] - 1).long()

    out = torch.zeros(
        (1, 1, x, y, z), device=embed.device, dtype=torch.int
    ).flatten()  # For indexing to work, the out tensor
    # has to be flat

    ind = torch.arange(
        0, x_ind.shape[-1], device=embed.device
    )  # For each out pixel, we take the embedding at
    # that loc and assign it to skeleton

    out[ind] = skeleton[
        :, :, x_ind, y_ind, z_ind
    ].int()  # assign the skeleton ind to the out tensor

    return out.view(1, 1, x, y, z)  # return the re-shaped out tensor


if __name__ == "__main__":
    import skimage.io as io

    img = io.imread("/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/plant-stem/train/0hrs_plant1_trim-acylYFP.tif")
    mask = io.imread("/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/plant-stem/train/0hrs_plant1_trim-acylYFP.labels.tif")
    skel = torch.load("/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/plant-stem/train/0hrs_plant1_trim-acylYFP.skeletons.trch")


    mask = torch.from_numpy(mask.astype(int)).contiguous().cuda()

    for k, v in skel.items():
        skel[k] = v.float().cuda()

    out = bake_skeleton(mask, skel, average=False)



    # n_mask = torch.zeros(len(skel)) # the size of each individual skeleton
    # ind_map = torch.zeros(len(skel)) # the mapping of unique id to index in skeleton
    #
    # max_shape = max(v.shape[0] for v in skel.values())
    #
    # all_skels = torch.zeros((len(skel), max_shape, 3))
    #
    # for i, (k,v) in enumerate(skel.items()):
    #     n_mask[i] = v.shape[0]
    #     ind_map[i] =  k
    #
    #     all_skels[i, 0:v.shape[0], :] = v
    #
    #
    # shape = img.shape
    # shape = (shape[1], shape[2], shape[0])
    # print(shape)
    #
