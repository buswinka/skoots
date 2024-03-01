from __future__ import annotations
import glob
import os.path
from typing import Dict
from typing import Tuple, Callable, List, Union, Optional
import math

import numpy as np
import skimage.io as io
import torch
from torch import Tensor
from torch.utils.data import Dataset
import logging

# from skoots.train.merged_transform import get_centroids
from tqdm import tqdm
from skoots.lib.custom_types import SparseDataDict

Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


class SparseDataloader(Dataset):
    def __init__(
        self,
        path: Union[List[str], str],
        transforms: Optional[Transform] = lambda x: x,
        device: Optional[str] = "cpu",
        sample_per_image: Optional[int] = 1,
    ):
        r"""
        Custom dataset for loading and accessing skoots training data. This class loads data based on filenames and
        specific extensions: '.tif' (raw image), '.labels.tif' (instance masks), '.skeletons.tif' (precomputed skeletons).
        An example training data folder might contain the following:
        ::
            data\
             └  train\
                  │ train_data.tif
                  │ train_data.background.tif
                  │ train_data.skeleton_mask.tif
                  └ train_data.skeletons.tif


        :param path: Path to training data
        :param transforms: A function which applies dataset augmentation on a data_dict
        :param pad_size: padding to add to every image in the dataset
        :param device: torch.device which to **output** all data on
        :param sample_per_image: number of times each image/mask pair is sampled per iteration over a dataset
        """

        super(Dataset, self).__init__()

        # Reassigning variables
        self.files: List[str] = []
        self.image: List[Tensor] = []
        self.background_mask: List[Tensor] = []
        self.skeletons: List[Dict[int, Tensor]] = []
        self.skeleton_mask: List[Tensor] = []
        self.transforms: Callable[[SparseDataDict], SparseDataDict] = transforms
        self.device = device

        self.sample_per_image: int = sample_per_image

        # Store all possible directories containing data in a list
        path: List[str] = [path] if isinstance(path, str) else path

        for p in path:
            self.files.extend(glob.glob(f"{p}{os.sep}*.background.tif"))

        for f in tqdm(self.files, desc="Loading Files: "):

            f = f.replace('.background.tif', '')
            if os.path.exists(f + ".tif"):
                image_path = f + ".tif"
                logging.info(f'Loading Image: {image_path}')
            else:
                raise FileNotFoundError(
                    f"Could not find file: {image_path[:-4:]} with extensions .tif"
                )

            skeleton = (
                torch.load(f + ".skeletons.trch")
                if os.path.exists(f + ".skeletons.trch")
                else {-1: torch.tensor([])}
            )
            assert -1 not in skeleton, f'Could not find a valid skeleton file for image: {f}.labels.tif'

            image: np.array = io.imread(image_path).astype(np.uint8)  # [Z, X, Y, C]

            skeleton_mask = io.imread(f + '.skeleton_mask.tif')
            background: np.ndarray = io.imread(f + '.background.tif')

            image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
            image: np.array = image.transpose(-1, 1, 2, 0)
            image: np.array = image[[2], ...] if image.shape[0] > 3 else image

            background: np.array = background.transpose(1, 2, 0).__gt__(0).astype(np.int32)
            assert background.max() == 1, f'{background.max()=}'
            skeleton_mask: np.array = skeleton_mask.transpose(1, 2, 0).astype(np.int32)

            assert image.max() < 256, f'16bit images not supported {image.max()=}'

            # Convert to torch.tensor
            image: Tensor = torch.from_numpy(image)  # .to(self.device)
            background: Tensor = (
                torch.from_numpy(background).gt(0).to(torch.uint8).unsqueeze(0)
            )  # .to(self.device)

            skeleton_mask: Tensor = (
                torch.from_numpy(skeleton_mask).gt(0).to(torch.uint8).unsqueeze(0)
            )  # .to(self.device)

            # I need the images in a float, but use torch automated mixed precision so can store as half.
            # This may not be the same for you!
            self.image.append(image)
            self.background_mask.append(background)
            self.skeleton_mask.append(skeleton_mask)

            for i, (k, v) in enumerate(skeleton.items()):
                if v.numel() == 0:
                    raise ValueError(f"{f} instance label {k} has {v.numel()=}. {skeleton.keys()=}")

            self.skeletons.append(skeleton)


    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> SparseDataDict:
        # We might artificially want to sample more times per image
        # Usefull when larging super large images with a lot of data.
        item = item // self.sample_per_image

        with torch.no_grad():
            data_dict: SparseDataDict = {
                "image": self.image[item],
                "background": self.background_mask[item],
                "skele_masks": self.skeleton_mask[item],
                "skeletons": self.skeletons[item],
            }

            # Transformation pipeline
            with torch.no_grad():
                data_dict: SparseDataDict = self.transforms(data_dict)  # Apply transforms

        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = data_dict[k].to(self.device)
            elif isinstance(data_dict[k], dict):
                data_dict[k] = {
                    key: value.to(self.device) for (key, value) in data_dict[k].items()
                }
        return data_dict

    def to(self, device: str) -> SparseDataloader:
        """
        Sends all data stored in the dataloader to a device.

        :param device: torch device for images, masks, and skeletons
        :return: self
        """
        self.image = [x.to(device) for x in self.image]
        self.background_mask = [x.to(device) for x in self.background_mask]
        self.skeleton_mask = [x.to(device) for x in self.skeleton_mask]
        self.skeletons = [
            {k: v.to(device) for (k, v) in x.items()} for x in self.skeletons
        ]

        return self

    def cuda(self) -> SparseDataloader:
        """alias for self.to('cuda:0')"""
        self.to("cuda:0")
        return self

    def cpu(self) -> SparseDataloader:
        """alias for self.to('cpu')"""
        self.to("cuda:0")
        self.to("cpu")
        return self

    def map_dd(self, fn: Callable[[SparseDataDict], SparseDataDict]) -> SparseDataloader:
        for i, (im, bg, sm, sk) in enumerate(zip(self.image, self.background_mask, self.skeleton_mask, self.skeletons)):
            data_dict: SparseDataDict = {
                "image": im,
                "background": bg,
                "skele_masks": sm,
                "skeletons": sk,
            }
            data_dict = fn(data_dict)
            self.image[i] = data_dict['image']
            self.background_mask[i] = data_dict['background']
            self.skeleton_mask[i] = data_dict['skele_masks']
            self.skeletons[i] = data_dict['skele_masks']

        return self

    def map(self, fn, key: List[str] | str) -> SparseDataloader:
        """
        applies a fn to an internal datastructure, provided by key.
        valid keys: ['image', 'background', 'skele_masks', 'skeletons']
        """
        _valid_keys = ['image', 'background', 'skele_masks', 'skeletons']
        key: List[str] = [key] if isinstance(key, str) else key
        for k in key:
            assert k in _valid_keys, f'key: {k} is invalid. Valid keys are: {_valid_keys}'

        logging.debug(f'attempting map operation on {self} with fn: {fn} and key: {key}')

        if 'image' in key:
            self.image = [fn(im) for im in self.image]
        if 'background' in key:
            self.background_mask = [fn(im) for im in self.background_mask]
        if 'skeletons' in key:
            self.skeletons = [fn(im) for im in self.skeletons]
        if 'skele_masks' in key:
            self.skeleton_mask = [fn(im) for im in self.skeleton_mask]

        return self

    def pin_memory(self) -> SparseDataloader:
        """
        Pins underlying memory allowing faster transfer to GPU
        """
        self.image = [x.pin_memory() for x in self.image]
        self.background_mask= [x.pin_memory() for x in self.background_mask]
        self.skeleton_mask= [x.pin_memory() for x in self.skeleton_mask]
        self.skeletons = [
            {k: v.pin_memory() for (k, v) in x.items()} for x in self.skeletons
        ]
        return self

    def sum(self, with_invert: bool = False) -> int:
        total = 0
        for x in self.image:
            total += x.cpu().sum()
        if with_invert:
            total += x.cpu().sub(255).mul(-1).sum()
        return total

    def numel(self, with_invert: bool = False) -> int:
        numel = sum([x.numel() for x in self.image]) if self.image else 0
        numel = numel * 2 if with_invert else numel
        return numel

    def mean(self, with_invert: bool = False) -> float | None:
        return self.sum(with_invert=with_invert) / self.numel(with_invert=with_invert)


    def std(self, with_invert: bool = False) -> float | None:
        mean = self.mean(with_invert=with_invert)
        n = self.numel(with_invert=with_invert)

        numerator = self.sum_subtract(mean) ** 2
        return math.sqrt(numerator / n)


    def subtract_square_sum(self, other) -> float:
        """
        returns the sum of the entire dataset, each px subtracted by other
        :param other:
        :return:
        """
        total = 0
        for x in self.image:
            total += x.cpu().to(torch.float64).sub(other).pow(2).sum()

        return total


def sparse_colate(
    data_dict: List[Dict[str, Tensor]]
) -> Tuple[Tensor, Tensor, List[Dict[str, Tensor]], Tensor, None]:
    """
    Colate function with defines how we batch training data.
    Unpacks a data_dict with keys: 'image', 'masks', 'skele_masks', 'baked_skeleton', 'skeleton'
    and puts them each into a Tensor. This should not be called outright, rather passed to a
    torch.DataLoader for automatic batching.

    :param data_dict: Dictonary of augmented training data
    :return: Tuple of batched data
    """
    images = torch.stack([dd.pop("image") for dd in data_dict], dim=0)
    background = torch.stack([dd.pop("background") for dd in data_dict], dim=0)
    skele_masks = torch.stack([dd.pop("skele_masks") for dd in data_dict], dim=0)

    skeletons = [dd.pop("skeletons") for dd in data_dict]

    return images, background, skeletons, skele_masks, None


if __name__ == '__main__':

    dataset = SparseDataloader(
        path='/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/mitochondria/sparse'
    )