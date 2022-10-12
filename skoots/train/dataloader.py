import torch
import torchvision.transforms.functional as ttf
from torch import Tensor
import numpy as np
import glob
import os.path
import skimage.io as io
from typing import Dict, List, Union
from torch.utils.data import Dataset
from hcat.train.merged_transform import get_centroids
from tqdm import tqdm
from typing import Tuple, Callable, List, Union, Optional

Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


class dataset(Dataset):
    def __init__(self,
                 path: Union[List[str], str],
                 transforms: Optional[Transform] = lambda x: x,
                 pad_size: Optional[int] = 100,
                 device: Optional[str] = 'cpu',
                 sample_per_image: Optional[int] = 1):

        super(Dataset, self).__init__()

        # Reassigning variables
        self.files = []
        self.image = []
        self.centroids = []
        self.masks = []
        self.skeletons = []
        self.baked_skeleton = []
        self.transforms = transforms
        self.device = device
        self.pad_size: List[int] = [pad_size, pad_size]

        self.sample_per_image = sample_per_image

        # Store all possible directories containing data in a list
        path: List[str] = [path] if isinstance(path, str) else path

        for p in path:
            self.files.extend(glob.glob(f'{p}{os.sep}*.labels.tif'))

        for f in tqdm(self.files, desc='Loading Files: '):
            if os.path.exists(f[:-11:] + '.tif'):
                image_path = f[:-11:] + '.tif'
            else:
                raise FileNotFoundError(f'Could not find file: {image_path[:-4:]} with extensions .tif')

            skeleton = torch.load(f[:-11:] + '.skeletons.trch') if os.path.exists(
                f[:-11:] + '.skeletons.trch') else {-1: torch.tensor([])}

            image: np.array = io.imread(image_path)  # [Z, X, Y, C]
            masks: np.array = io.imread(f)  # [Z, X, Y]

            image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
            image: np.array = image.transpose(-1, 1, 2, 0)
            image: np.array = image[[2], ...] if image.shape[0] > 3 else image

            masks: np.array = masks.transpose(1, 2, 0).astype(np.int32)

            scale: int = 2 ** 16 if image.max() > 256 else 255  # Our images might be 16 bit, or 8 bit
            scale = scale if image.max() > 1 else 1.

            # Convert to torch.tensor
            image: Tensor = torch.from_numpy(image / scale).to(self.device)
            masks: Tensor = torch.from_numpy(masks).int().unsqueeze(0).to(self.device)

            # I need the images in a float, but use torch automated mixed precision so can store as half.
            # This may not be the same for you!
            self.image.append(image.half())
            self.masks.append(masks)


            for i, (k, v) in enumerate(skeleton.items()):
                if v.numel() == 0:
                    skeleton[k] = centroids[i, :].unsqueeze(0)

            skeleton = {int(k): v for k, v in skeleton.items()}


            self.skeletons.append(skeleton)
            self.baked_skeleton.append(None)

    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        # We might artificially want to sample more times per image
        # Usefull when larging super large images with a lot of data.
        item = item // self.sample_per_image

        with torch.no_grad():
            data_dict = {'image': self.image[item],
                         'masks': self.masks[item],
                         'skeletons': self.skeletons[item],
                         'baked-skeleton': self.baked_skeleton[item]}

            # Transformation pipeline
            with torch.no_grad():
                data_dict = self.transforms(data_dict)  # Apply transforms

        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = data_dict[k].to(self.device)
            elif isinstance(data_dict[k], dict):
                data_dict[k] = {key: value.to(self.device) for (key, value) in data_dict[k].items()}

        return data_dict

    def to(self, device: str):
        """
        It is faster to do transforms on cuda, and if your GPU is big enough, everything can live there!!!
        """
        self.image = [x.to(device) for x in self.image]
        self.masks = [x.to(device) for x in self.masks]
        self.skeletons = [{k: v.to(device) for (k, v) in x.items()} for x in self.skeletons]

        return self

    def cuda(self):
        self.to('cuda:0')
        return self

    def cpu(self):
        self.to('cpu')
        return self


class BackgroundDataset(Dataset):
    def __init__(self,
                 path: Union[List[str], str],
                 transforms: Optional[Transform] = lambda x: x,
                 pad_size: Optional[int] = 100,
                 device: Optional[str] = 'cpu',
                 sample_per_image: Optional[int] = 1):
        super(Dataset, self).__init__()
        """
        A dataset for images that contain nothing
        """

        # Reassigning variables
        self.files = []
        self.image = []
        self.transforms = transforms
        self.device = device

        path: List[str] = [path] if isinstance(path, str) else path

        for p in path:
            self.files.extend(glob.glob(f'{p}{os.sep}*.labels.tif'))

        for f in tqdm(self.files, desc='Loading Files: '):
            if os.path.exists(f[:-11:] + '.tif'):
                image_path = f[:-11:] + '.tif'
            else:
                raise FileNotFoundError(f'Could not find file: {image_path[:-4:]} with extensions .tif')

            skeleton = torch.load(f[:-11:] + '.skeletons.trch') if os.path.exists(
                f[:-11:] + '.skeletons.trch') else {-1: torch.tensor([])}

            image: np.array = io.imread(image_path)  # [Z, X, Y, C]
            masks: np.array = io.imread(f)  # [Z, X, Y]

            image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
            image: np.array = image.transpose(-1, 1, 2, 0)
            image: np.array = image[[2], ...] if image.shape[0] > 3 else image

            scale: int = 2 ** 16 if image.max() > 256 else 255  # Our images might be 16 bit, or 8 bit
            scale: int = scale if image.max() > 1 else 1

            image: Tensor = torch.from_numpy(image / scale).to(self.device)
            self.image.append(image.half())

    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        # We might artificially want to sample more times per image
        # Usefull when larging super large images with a lot of data.
        item = item // self.sample_per_image

        with torch.no_grad():
            data_dict = {'image': self.image[item],
                         'masks': torch.empty((1)),
                         'skeletons': {-1: torch.empty((1))},
                         'baked-skeleton': None}

            # Transformation pipeline
            with torch.no_grad():
                data_dict = self.transforms(data_dict)  # Apply transforms

        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = data_dict[k].to(self.device)
            elif isinstance(data_dict[k], dict):
                data_dict[k] = {key: value.to(self.device) for (key, value) in data_dict[k].items()}

        return data_dict

    def to(self, device: str):
        """
        It is faster to do transforms on cuda, and if your GPU is big enough, everything can live there!!!
        """
        self.image = [x.to(device) for x in self.image]
        self.masks = [x.to(device) for x in self.masks]
        self.skeletons = [{k: v.to(device) for (k, v) in x.items()} for x in self.skeletons]

        return self

    def cuda(self):
        self.to('cuda:0')
        return self

    def cpu(self):
        self.to('cpu')
        return self


class MultiDataset(Dataset):
    def __init__(self, *args):
        self.datasets: List[Dataset] = []
        for ds in args:
            if isinstance(ds, Dataset):
                self.datasets.append(ds)

        self._dataset_lengths = [len(ds) for ds in self.datasets]

        self.num_datasets = len(self.datasets)

        self._mapped_indicies = []
        for i, ds in enumerate(self.datasets):
            # range(len(ds)) necessary to not index whole dataset at start. SLOW!!!
            self._mapped_indicies.extend([i for _ in range(len(ds))])

    def __len__(self):
        return len(self._mapped_indicies)

    def __getitem__(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offset
        try:
            return self.datasets[i][item - _offset]
        except Exception:
            print(i, _offset, item - _offset, item, len(self.datasets[i]), self.datasets[i].files[item])
            raise RuntimeError

    def to(self, device: str):
        for i in range(self.num_datasets):
            self.datasets[i].to(device)
        return self

    def cuda(self):
        for i in range(self.num_datasets):
            self.datasets[i].to('cuda:0')
        return self

    def cpu(self):
        for i in range(self.num_datasets):
            self.datasets[i].to('cpu')
        return self



# Custom batching function!
def skeleton_colate(data_dict: List[Dict[str, Tensor]]) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    images = torch.stack([dd.pop('image') for dd in data_dict], dim=0)
    masks = torch.stack([dd.pop('masks') for dd in data_dict], dim=0)
    skele_masks = torch.stack([dd.pop('skele_masks') for dd in data_dict], dim=0)
    baked = [dd.pop('baked-skeleton') for dd in data_dict]

    if baked[0] is not None:
        baked = torch.stack(baked, dim=0)

    skeletons = [dd.pop('skeletons') for dd in data_dict]

    return images, masks, skeletons, skele_masks, baked


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from hcat.train.merged_transform import merged_transform_3D
    from torch.utils.data import DataLoader

    path = '/home/chris/Dropbox (Partners HealthCare)/HairCellInstance/data/segmentation/'

    data = dataset(path=path + 'train',
                   transforms=merged_transform_3D, device='cuda:0')

    dl = DataLoader(data, batch_size=2, shuffle=False, collate_fn=colate)
    for _ in range(1):
        for i, (image, masks, centroids) in enumerate(dl):
            print(image.max())
            plt.imshow(image[0, 0, :, :, 7].cpu().float().numpy())
            plt.imshow(masks[0, 0, :, :, 7].gt(0).cpu().float().numpy(), alpha=0.2)

            for c in centroids[0]:
                plt.plot(c[1].cpu().numpy(), c[0].cpu().numpy(), 'ro')
            plt.title(i)
            plt.show()
            if centroids[0].numel() == 0:
                raise ValueError
