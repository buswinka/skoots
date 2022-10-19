import torch
import torch.nn.functional as F
from torch import Tensor
from skoots.lib.embedding_to_prob import baked_embed_to_prob
from skoots.lib.vector_to_embedding import vector_to_embedding
from skoots.lib.flood_fill import efficient_flood_fill
from skoots.lib.cropper import crops
from skoots.lib.skeleton import bake_skeleton
from skoots.lib.morphology import binary_erosion, binary_dilation

import skimage.io as io
from tqdm import tqdm
import numpy as np
from torch import Tensor

from bism.models import get_constructor
from bism.models.spatial_embedding import SpatialEmbedding
from torch.cuda.amp import GradScaler, autocast

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from skimage.morphology import skeletonize

"""
Instance Segmentation more or less...
---------------------
    vectors, skeletons, masks = model(image)
    instance_skeletons = efficient_flood_fill(skeletons)
    for id in instance_skeletons.unique():
        instance_mask = get_instance(mask, vectors, instance_skeletons)

"""


# image_path = '/home/chris/Documents/threeOHC_registered-scaled.tif'
# image_path = '/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Figure 1 - overview/data/single_mito.tif'
# image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/data/validation/hide-1_150-201.tif'
# image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/data/test/hide001.tif'

@torch.no_grad()
def get_instance(mask: Tensor,
                 vectors: Tensor,
                 skeleton: Tensor,
                 id: int,
                 num: Tensor,
                 thr: float = 0.5,
                 sigma: Tensor = torch.tensor((10, 10, 3)),
                 min_instance_volume: int = 183) -> Tensor:
    """
    Gets an instance mask of a single object associated with identified Skeleton


    Shapes:
        - mask: :math:`(1, X_{in}, Y_{in}, Z_{in})`
        - vectors: :math:`(3, X_{in}, Y_{in}, Z_{in})`
        - skeleton: :math:`(3, N_{j})`

    :param mask: empty mask by which to store all instances
    :param vectors: Embedding Vectors predicted by a neural network with shape [C=3, X, Y, Z]
    :param skeleton: Tensor of pixels representing the skeleton of an instance with shape [C=3, N] from *efficient_flood_fill*
    :param id: ID value of skeleton of interest
    :param num: anisotropic scaling factors
    :param thr: instance probability threshold
    :param min_instance_volume: rejects instances smaller than this param

    :return: Semantic mask of the individual instance
    """

    # Establish min and max indicies
    buffer = torch.tensor([50, 50, 10], device=skeleton.device)  # on all sides of the skeleton
    ind_min = (skeleton - buffer).clamp(0)
    ind_max = skeleton + buffer

    for i in range(3):  # Clamp this to the vindow...
        ind_max[:, i] = ind_max[:, i].clamp(0, vectors.shape[i + 1])  # Vector is [3, X, Y, Z]

    ind_min = ind_min.min(0)[0]
    ind_max = ind_max.max(0)[0]

    # Get a crop of the vectors
    crop = vectors[:,
           ind_min[0]:ind_max[0],
           ind_min[1]:ind_max[1],
           ind_min[2]:ind_max[2]].unsqueeze(0).cuda()

    mask_crop = mask[
                ind_min[0]:ind_max[0],
                ind_min[1]:ind_max[1],
                ind_min[2]:ind_max[2]].unsqueeze(0).cuda()

    # for i in range(crop.shape[-1]):
    #     plt.imshow(mask_crop[0, :, :, i].detach().cpu().numpy())
    #     plt.title(f'maks_{i}')
    #     plt.show()
    #
    #     plt.imshow(crop[0, 0, :, :, i].detach().cpu().numpy())
    #     plt.title(f'vec_{i}')
    #     plt.show()

    crop = vector_to_embedding(scale=num.cuda(), vector=crop)

    skeleton = skeleton.sub(ind_min)  # Adjust skeleton and put into a dict

    baked = bake_skeleton(masks=mask_crop.add(1).gt(0).cpu(), skeletons={1: skeleton}, average=True, device='cpu')

    # Get the probability from the skeleton

    prob = baked_embed_to_prob(crop, baked.to(crop.device), sigma=sigma.to(crop.device))[0]
    prob = prob.gt(thr).mul(id).squeeze()

    # Put it back into the mask
    index = prob == id
    if torch.sum(index) > min_instance_volume:
        mask[
        ind_min[0]:ind_max[0],
        ind_min[1]:ind_max[1],
        ind_min[2]:ind_max[2],
        ][index] = prob[index].cpu().float()
    else:
        mask[
        ind_min[0]:ind_max[0],
        ind_min[1]:ind_max[1],
        ind_min[2]:ind_max[2],
        ][index] = 0

    return mask


def eval(image_path: str) -> None:
    scale = -99999

    image: np.array = io.imread(image_path)  # [Z, X, Y, C]
    image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
    image: np.array = image.transpose(-1, 1, 2, 0)
    image: np.array = image[[2], ...] if image.shape[0] > 3 else image  # [C=1, X, Y, Z]

    if image.max() > 256:
        scale: int = 2 ** 16
    elif image.max() <= 256 and image.max() > 1:
        scale = 256
    elif image.max() < 1 and image.max() > 0:
        scale = 1

    image: Tensor = torch.from_numpy(image / scale)

    print(image.shape, scale, image.max(), image.min())

    # image = image.transpose(0, -1).squeeze().unsqueeze(0)
    # image = torch.clamp(image, 0, 1)
    pad3d = (5, 5, 30, 30, 30, 30)  # Pads last dim first!
    # pad3d = False
    image = F.pad(image, pad3d, mode='reflect')
    print(image.shape)

    num_tuple = (60, 60, 2)
    num = torch.tensor(num_tuple)

    #
    # image = torch.load('/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Figure 1 - overview/data/images.trch')
    # image = image[1, ...]
    # io.imsave('/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Figure 1 - overview/data/' +
    #           'test.tif', image.permute(3,1,2,0).numpy())
    # # return

    c, x, y, z = image.shape

    skeleton = torch.zeros(size=(1, x, y, z))
    semantic = torch.zeros((1, x, y, z))
    vectors = torch.zeros((3, x, y, z))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(
        '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/models/Oct14_13-09-30_CHRISUBUNTU.trch')

    state_dict = checkpoint if not 'model_state_dict' in checkpoint else checkpoint['model_state_dict']

    model_constructor = get_constructor('unext', spatial_dim=3)  # gets the model from a name...
    backbone = model_constructor(in_channels=1, out_channels=60, dims=[60, 120, 240, 120, 60])
    model = SpatialEmbedding(
        backbone=backbone
    )
    model.load_state_dict(state_dict=state_dict)

    model = model.to(device).train()
    cropsize = [300, 300, 20]
    overlap = [60, 60, 5]

    image = image.float()
    print(image.shape)

    print('Begining Analysis....', end='')
    iterator = tqdm(crops(image, cropsize, overlap), desc='')

    id = 1

    with torch.no_grad():
        for slice, (x, y, z) in iterator:
            with autocast(enabled=True):  # Saves Memory!
                out = model(slice.float().cuda())

            probability_map = out[:, [-1], ...].cpu()
            skeleton_map = out[:, [-2], ...].cpu()
            vec = out[:, 0:3:1, ...].cpu()

            skeleton[:,
            x + overlap[0]: x + cropsize[0] - overlap[0],
            y + overlap[1]: y + cropsize[1] - overlap[1],
            z + overlap[2]: z + cropsize[2] - overlap[2]] = skeleton_map[0, :,
                                                            overlap[0]: -overlap[0],
                                                            overlap[1]: -overlap[1],
                                                            overlap[2]: -overlap[2]
                                                            :].gt(0.5)

            vectors[:,
            x + overlap[0]: x + cropsize[0] - overlap[0],
            y + overlap[1]: y + cropsize[1] - overlap[1],
            z + overlap[2]: z + cropsize[2] - overlap[2]] = vec[0, :,
                                                            overlap[0]: -overlap[0],
                                                            overlap[1]: -overlap[1],
                                                            overlap[2]: -overlap[2]
                                                            :]
            semantic[:,
            x + overlap[0]: x + cropsize[0] - overlap[0],
            y + overlap[1]: y + cropsize[1] - overlap[1],
            z + overlap[2]: z + cropsize[2] - overlap[2]] = probability_map[0, :,
                                                            overlap[0]: -overlap[0],
                                                            overlap[1]: -overlap[1],
                                                            overlap[2]: -overlap[2]
                                                            :]

            iterator.desc = f'Evaluating slice at: [x{x}:y{y}:z{z}]'

        # _x, _y, _z
        if pad3d:
            skeleton = skeleton[0, pad3d[2]:-pad3d[3], pad3d[4]:-pad3d[5], pad3d[0]:-pad3d[1]]
            semantic = semantic[0, pad3d[2]:-pad3d[3], pad3d[4]:-pad3d[5], pad3d[0]:-pad3d[1]]
            vectors = vectors[:, pad3d[2]:-pad3d[3], pad3d[4]:-pad3d[5], pad3d[0]:-pad3d[1]]

        else:
            skeleton = skeleton[0, ...]
            semantic = semantic[0, ...]
            vectors = vectors[:, ...]

        id = 2

        skeleton = skeleton.mul(semantic.gt(0.5))
        vectors = vectors * semantic.gt(0.5)

        # for i in range(1):
        #     skeleton = dilate(skeleton)
        print(f'Semanntic Mask\n\t{semantic.max()=}\n\t{semantic.min()=}')

        io.imsave('/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/semantic.tif',
                  semantic.mul(255).round().int().cpu().numpy().astype(np.uint8).transpose(2, 0, 1))

        io.imsave(
            '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/skeleton_unlabeled.tif',
            skeleton.squeeze().cpu().numpy().astype(np.uint16).transpose(2, 0, 1))

        io.imsave('/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/vectors.tif',
                  vectors.mul(2).div(2).mul(255).round().cpu().numpy().transpose(-1, 1, 2, 0))

        print('Saved', skeleton.shape, semantic.shape)

        skeleton, skeleton_dict = efficient_flood_fill(skeleton, skeletonize=True, device='cuda:0')

        io.imsave('/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/skeleton.tif',
                  skeleton.cpu().numpy().astype(np.uint16).transpose(2, 0, 1))

    instance_mask = torch.zeros_like(semantic).cpu()

    for k in tqdm(skeleton_dict.keys()):
        instance_mask = get_instance(instance_mask, vectors, skeleton_dict[k], k, num)

    print(instance_mask.unique().shape[0] - 1, ' Unique mito')

    io.imsave('/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/instance_mask.tif',
              instance_mask.cpu().numpy().astype(np.uint16).transpose(2, 0, 1))


if __name__ == '__main__':
    image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/hide_validate-1.tif'
    eval(image_path)
