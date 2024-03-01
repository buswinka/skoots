from __future__ import annotations
import glob
import os.path
from typing import Dict
from typing import Tuple, Callable, List, Union, Optional, Any
import math

import numpy as np
import skimage.io as io
import torch
from torch import Tensor
from torch.utils.data import Dataset
import logging

# from skoots.train.merged_transform import get_centroids
from skoots.lib.custom_types import DataDict
import torch.jit
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _sub_sq_sum(x: np.ndarray, other):
    """
    subtracts other, sqares the result for each val, and sums to one number.

    :param x: np.ndarray
    :param other: number
    :return: number
    """

    total = np.array([0.0], dtype=np.float64)
    x = x.flatten()
    for i in prange(x.shape[0]):
        total += (float(x[i]) - other) ** 2
    return total


Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


class dataset(Dataset):
    def __init__(
        self,
        path: Union[List[str], str],
        transforms: Optional[Transform] = lambda x: x,
        pad_size: Optional[int] = 100,
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
                  │ train_data.labels.tif
                  └ train_data.skeletons.tif


        :param path: Path to training data
        :param transforms: A function which applies dataset augmentation on a data_dict
        :param pad_size: padding to add to every image in the dataset
        :param device: torch.device which to **output** all data on
        :param sample_per_image: number of times each image/mask pair is sampled per iteration over a dataset
        """

        super(Dataset, self).__init__()

        # Reassigning variables
        self.path = path
        self.files: List[str] = []
        self.image: List[Tensor] = []
        self.centroids: List[Tensor] = []
        self.masks: List[Tensor] = []
        self.skeletons: List[Dict[int, Tensor]] = []
        self.baked_skeleton: List[Tensor] = []
        self.transforms: Callable[[DataDict], DataDict] = transforms
        self.device = device
        self.pad_size: List[int] = [pad_size, pad_size]

        # cached mean
        self.__mean: Dict[str, Any] | None = None
        self.__std: Dict[str, Any] | None = None
        self.__numel: Dict[str, Any] | None = None
        self.__sum: Dict[str, Any] | None = None

        self.sample_per_image: int = sample_per_image

        # Store all possible directories containing data in a list
        path: List[str] = [path] if isinstance(path, str) else path
        logging.debug(f"Attempting to load files from: {path}")

        for p in path:
            self.files.extend(glob.glob(f"{p}{os.sep}*.labels.tif"))

        logging.debug(f"found the following files to load:")
        for f in self.files:
            logging.debug(f"----> {f}")

        for f in self.files:
            if os.path.exists(f[:-11:] + ".tif"):
                image_path = f[:-11:] + ".tif"
            else:
                raise FileNotFoundError(
                    f"Could not find file: {image_path[:-4:]} with extensions .tif"
                )
            assert os.path.exists(f.replace('.labels.tif', '.skeletons.trch')), f'cannot find skeleton file for: {f}'

            skeleton = (
                torch.load(f[:-11:] + ".skeletons.trch")
                if os.path.exists(f[:-11:] + ".skeletons.trch")
                else {-1: torch.tensor([])}
            )

            logging.info(f"Loading Image: {image_path}")
            image: np.array = io.imread(image_path)
            masks: np.array = io.imread(f)  # [Z, X, Y]

            assert image.dtype == np.uint8, f"image must be 8bit"

            image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
            image: np.array = image.transpose(-1, 1, 2, 0)
            image: np.array = image[[2], ...] if image.shape[0] > 3 else image

            masks: np.array = masks.transpose(1, 2, 0)

            if masks.max() < 256:
                logging.info(f"saving mask at {f} as dtype: uint8")
                masks = masks.astype(np.uint8)
            elif masks.max() < (2**16 // 2) - 1:
                logging.info(f"saving mask at {f} as dtype: int16")
                masks = masks.astype(np.int16)
            else:
                logging.info(f"saving mask at {f} as dtype: int32")
                masks = masks.astype(np.int32)

            # Convert to torch.tensor
            image: Tensor = torch.from_numpy(image)  # .to(self.device)
            masks: Tensor = torch.from_numpy(masks).unsqueeze(0)

            # I need the images in a float, but use torch automated mixed precision so can store as half.
            # This may not be the same for you!
            self.image.append(image)
            self.masks.append(masks)



            for i, (k, v) in enumerate(skeleton.items()):
                if v.numel() == 0:
                    raise ValueError(f"{f} instance label {k} has {v.numel()=}")

            self.skeletons.append(skeleton)
            self.baked_skeleton.append(None)

        logging.info(f"done loading from source: {path}")

    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> DataDict:
        # We might artificially want to sample more times per image
        # Usefull when larging super large images with a lot of data.
        item = item // self.sample_per_image

        with torch.no_grad():
            data_dict: DataDict = {
                "image": self.image[item],
                "masks": self.masks[item],
                "skeletons": self.skeletons[item],
                "baked_skeleton": self.baked_skeleton[item],
            }

            # Transformation pipeline
            with torch.no_grad():
                data_dict: DataDict = self.transforms(data_dict)  # Apply transforms

        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = data_dict[k].to(self.device)
            elif isinstance(data_dict[k], dict):
                data_dict[k] = {
                    key: value.to(self.device) for (key, value) in data_dict[k].items()
                }

        return data_dict

    def to(self, device: str):
        """
        Sends all data stored in the dataloader to a device.

        :param device: torch device for images, masks, and skeletons
        :return: self
        """
        self.image = [x.to(device) for x in self.image]
        self.masks = [x.to(device) for x in self.masks]
        self.skeletons = [
            {k: v.to(device) for (k, v) in x.items()} for x in self.skeletons
        ]

        return self

    def cuda(self) -> dataset:
        """alias for self.to('cuda:0')"""
        self.to("cuda:0")
        return self

    def cpu(self) -> dataset:
        """alias for self.to('cpu')"""
        self.to("cuda:0")
        self.to("cpu")
        return self

    def pin_memory(self) -> dataset:
        """
        Pins underlying memory allowing faster transfer to GPU
        """
        self.image = [x.pin_memory() for x in self.image]
        self.masks = [x.pin_memory() for x in self.masks]
        self.skeletons = [
            {k: v.pin_memory() for (k, v) in x.items()} for x in self.skeletons
        ]
        return self

    def map(self, fn, key: List[str] | str) -> BackgroundDataset:
        """
        applies a fn to an internal datastructure, provided by key.
        valid keys: ['image', 'background', 'skele_masks', 'skeletons']
        """
        _valid_keys = ["image", "masks", "skeletons"]
        key: List[str] = [key] if isinstance(key, str) else key
        for k in key:
            assert (
                k in _valid_keys
            ), f"key: {k} is invalid. Valid keys are: {_valid_keys}"

        if key == "image":
            self.image = [fn(im) for im in self.image]
        if key == "masks":
            self.masks = [fn(im) for im in self.masks]
        if key == "skeletons":
            self.skeletons = [fn(im) for im in self.skeletons]
        return self

    def sum(self, with_invert: bool = False) -> int:
        if self.__sum is None or self.__sum["with_invert"] != with_invert:
            logging.info(
                f"calculating sum of dataset from: {self.path} | {with_invert=}"
            )
            total = 0
            for x in self.image:
                total += x.cpu().sum()
            if with_invert:
                total += x.cpu().sub(255).mul(-1).sum()

            self.__sum = {"sum": total, "with_invert": with_invert}
        return self.__sum["sum"]

    def numel(self, with_invert: bool = False) -> int:
        if self.__numel is None or self.__numel["with_invert"] != with_invert:
            logging.info(
                f"calculating numel of dataset from: {self.path} | {with_invert=}"
            )
            numel = sum([x.numel() for x in self.image]) if self.image else 0
            numel = numel * 2 if with_invert else numel
            self.__numel = {"numel": numel, "with_invert": with_invert}

        return self.__numel["numel"]

    def mean(self, with_invert: bool = False) -> float | None:
        if self.__mean is None or self.__mean["with_invert"] != with_invert:
            logging.info(
                f"calculating mean of dataset from: {self.path} | {with_invert=}"
            )
            self.__mean = {
                "mean": self.sum(with_invert=with_invert)
                / self.numel(with_invert=with_invert),
                "with_invert": with_invert,
            }
        return self.__mean["mean"]

    def std(self, with_invert: bool = False) -> float | None:
        if self.__std is None or self.__std["with_invert"] != with_invert:
            logging.info(
                f"calculating std of dataset from: {self.path} | {with_invert=}"
            )
            mean = self.mean(with_invert=with_invert)
            n = self.numel(with_invert=with_invert)

            numerator = self.subtract_square_sum(mean) ** 2
            self.__std = {"std": math.sqrt(numerator / n), "with_invert": with_invert}

        return self.__std["std"]

    def subtract_square_sum(self, other):
        """
        returns the sum of the entire dataset, each px subtracted by other
        :param other:
        :return:
        """
        logging.info(
            f"performing substract_square_sum on dataset {self.path} | {other=}"
        )
        with torch.no_grad():
            total = 0
            for x in self.image:
                total += _sub_sq_sum(x.cpu().numpy(), other)

        return total



class BackgroundDataset(Dataset):
    def __init__(
        self,
        path: Union[List[str], str],
        transforms: Optional[Transform] = lambda x: x,
        device: Optional[str] = "cpu",
        sample_per_image: Optional[int] = 1,
    ):
        super(Dataset, self).__init__()
        r"""
        Custom dataset for loading and accessing skoots background training data. 
        Unlike skoots.train.dataloader.dataset, which looks for masks and skeletons, 
        this dataset meed only images given that the images do not contain any actuall instances of the thing you're
        trying to segment - i.e. its background. 
        
        An example training data folder might contain the following:
        ::
            data\
             └  background\
                  └ background_image.tif


        :param path: Path to background data
        :param transforms: A function which applies background_dataset augmentation on a data_dict
        :param pad_size: padding to add to every image in the dataset
        :param device: torch.device which to **output** all data on
        :param sample_per_image: number of times each image/mask pair is sampled per iteration over a dataset
           """

        # Reassigning variables
        self.files = []
        self.image = []
        self.transforms = transforms
        self.device = device
        self.sample_per_image = sample_per_image

        path: List[str] = [path] if isinstance(path, str) else path

        for p in path:
            self.files.extend(glob.glob(f"{p}{os.sep}*.labels.tif"))

        for f in self.files:
            if os.path.exists(f[:-11:] + ".tif"):
                image_path = f[:-11:] + ".tif"
            else:
                raise FileNotFoundError(
                    f"Could not find file: {image_path[:-4:]} with extensions .tif"
                )

            skeleton = (
                torch.load(f[:-11:] + ".skeletons.trch")
                if os.path.exists(f[:-11:] + ".skeletons.trch")
                else {-1: torch.tensor([])}
            )

            image: np.array = io.imread(image_path)  # [Z, X, Y, C]
            masks: np.array = io.imread(f)  # [Z, X, Y]

            image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
            image: np.array = image.transpose(-1, 1, 2, 0)
            image: np.array = image[[2], ...] if image.shape[0] > 3 else image

            scale: int = (
                2**16 if image.max() > 256 else 255
            )  # Our images might be 16 bit, or 8 bit
            scale: int = scale if image.max() > 1 else 1

            assert image.max() < 256, "16bit images not supported"
            image: Tensor = torch.from_numpy(image.astype(np.uint8))  # .to(self.device)

            self.image.append(image)

    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        # We might artificially want to sample more times per image
        # Usefull when larging super large images with a lot of data.
        item = item // self.sample_per_image

        with torch.no_grad():
            data_dict = {
                "image": self.image[item],
                "masks": torch.empty((1)),
                "skeletons": {-1: torch.empty((1))},
                "baked_skeleton": None,
            }

            # Transformation pipeline
            with torch.no_grad():
                data_dict = self.transforms(data_dict)  # Apply transforms

        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = data_dict[k].to(self.device, non_blocking=True)
            elif isinstance(data_dict[k], dict):
                data_dict[k] = {
                    key: value.to(self.device, non_blocking=True)
                    for (key, value) in data_dict[k].items()
                }

        return data_dict

    def to(self, device: str):
        """
        Sends all data stored in the dataloader to a device.

        :param device: torch device for images, masks, and skeletons
        :return: self
        """
        self.image = [x.to(device) for x in self.image]
        self.masks = [x.to(device) for x in self.masks]
        self.skeletons = [
            {k: v.to(device) for (k, v) in x.items()} for x in self.skeletons
        ]

        return self

    def cuda(self):
        """alias for self.to('cuda:0')"""
        self.to("cuda:0")
        return self

    def cpu(self):
        """alias for self.to('cpu')"""
        self.to("cpu")
        return self

    def map(self, fn, key: List[str] | str) -> BackgroundDataset:
        """
        applies a fn to an internal datastructure, provided by key.
        valid keys: ['image', 'background', 'skele_masks', 'skeletons']
        """
        _valid_keys = ["image"]
        key: List[str] = [key] if isinstance(key, str) else key
        for k in key:
            assert (
                k in _valid_keys
            ), f"key: {k} is invalid. Valid keys are: {_valid_keys}"

        if key == "image":
            self.image = [fn(im) for im in self.image]

        return self

    def sum(self):
        total = 0
        for x in self.image:
            total += x.sum()
        return total

    def numel(self):
        numel = sum([x.numel() for x in self.image]) if self.image else 0
        return numel

    def mean(self):
        logging.debug(f"Calculating dataset mean for {self}")
        if self.image:
            return self.sum() / self.numel
        else:
            return None

    def std(self):
        logging.debug(f"Calculating dataset mean for {self}")
        mean = self.mean()
        n = self.numel()

        if mean is not None:
            numerator = self.sum_subtract(mean) ** 2
            return math.sqrt(numerator / n)
        else:
            return None

    def subtract_square_sum(self, other):
        """
        returns the sum of the entire dataset, each px subtracted by other
        :param other:
        :return:
        """
        total = 0
        for x in self.image:
            total += x.to(torch.float64).sub(other).pow(2).sum()

        return total


class MultiDataset(Dataset):
    def __init__(self, *args):
        r"""
        A utility class for joining multiple datasets into one accessible class. Sometimes, you may subdivide your
        training data based on some criteria. The most common is size: data from folder data/train/train_alot must be sampled 100 times
        per epoch, while data from folder data/train/train_notsomuch might only want to be sampled 1 times per epoch.

        You could construct a two skoots.train.dataloader.dataset objects for each
        and access both in a single MultiDataset class...

        >>> from skoots.train.dataloader import dataset
        >>>
        >>> # has one image sampled 100 times
        >>> data0 = dataset('data/train/train_alot', sample_per_image=100)
        >>> print(len(data0))  # 100
        >>>
        >>> # has one image sampled once
        >>> data1 = dataset('data/train/train_notsomuch', sample_per_image=1)
        >>> print(len(data1))  # 1
        >>>
        >>> merged_data = MultiDataset(data0, data1)
        >>> print(len(merged_data))  # 101, they've been merged!

        :param args:
        :type args:
        """
        self.datasets: List[dataset] = []
        for ds in args:
            if isinstance(ds, Dataset):
                self.datasets.append(ds)

        self._dataset_lengths = [len(ds) for ds in self.datasets]

        self.num_datasets = len(self.datasets)

        self._mapped_indicies = []
        for i, ds in enumerate(self.datasets):
            # range(len(ds)) necessary to not index whole dataset at start. SLOW!!!
            self._mapped_indicies.extend([i for _ in range(len(ds))])

    def __len__(self) -> int:
        return len(self._mapped_indicies)

    def __getitem__(self, item: int) -> DataDict:
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offset
        try:
            return self.datasets[i][item - _offset]
        except Exception as e:
            print(i, _offset, item - _offset, item, len(self.datasets[i]))
            raise e

    def to(self, device: str) -> MultiDataset:
        """
        Sends all data stored in the dataloader to a device. Occurs for ALL wrapped datasets.

        :param device: torch device for images, masks, and skeletons
        :return: self
        """
        for i in range(self.num_datasets):
            self.datasets[i].to(device)
        return self

    def cuda(self) -> MultiDataset:
        """alias for self.to('cuda:0')"""
        for i in range(self.num_datasets):
            self.datasets[i].to("cuda:0")
        return self

    def cpu(self) -> MultiDataset:
        """alias for self.to('cpu')"""
        for i in range(self.num_datasets):
            self.datasets[i].to("cpu")
        return self

    def map(self, fn, key) -> MultiDataset:
        for i in range(self.num_datasets):
            self.datasets[i].map(fn, key)  # ocurs in place
        return self

    def sum(self, with_invert: bool = False):
        logging.debug(f"calculating dataset sum for {self} | {with_invert=}")
        total = 0
        for d in self.datasets:
            _sum = d.sum(with_invert=with_invert)
            total = total + _sum if _sum is not None else total

        if total:
            return total
        else:
            return None

    def numel(self, with_invert: bool = False):
        logging.debug(f"calculating dataset numel for {self} | {with_invert=}")
        total = 0
        for d in self.datasets:
            _numel = d.numel(with_invert=with_invert)
            total = total + _numel if _numel is not None else total

        if total:
            return total
        else:
            return None

    def mean(self, with_invert: bool = False):
        logging.debug(f"calculating dataset mean for {self} | {with_invert=}")
        _sum = self.sum(with_invert=with_invert)
        _numel = self.numel(with_invert=with_invert)

        if _sum and _numel:
            return _sum / _numel
        else:
            return None

    def std(self, with_invert: bool = False):
        logging.debug(f"calculating dataset std for {self} | {with_invert=}")
        mean = float(self.mean(with_invert=with_invert))
        n = self.numel(with_invert=with_invert)

        if mean is not None:
            numerator = sum([d.subtract_square_sum(mean) for d in self.datasets])
            return math.sqrt(numerator / n)
        else:
            return None


# Custom batching function!
def skeleton_colate(
    data_dict: List[Dict[str, Tensor]]
) -> Tuple[Tensor, Tensor, List[Dict[str, Tensor]], Tensor, Tensor]:
    """
    Colate function with defines how we batch training data.
    Unpacks a data_dict with keys: 'image', 'masks', 'skele_masks', 'baked_skeleton', 'skeleton'
    and puts them each into a Tensor. This should not be called outright, rather passed to a
    torch.DataLoader for automatic batching.

    :param data_dict: Dictonary of augmented training data
    :return: Tuple of batched data
    """
    images = torch.stack([dd.pop("image") for dd in data_dict], dim=0)
    masks = torch.stack([dd.pop("masks") for dd in data_dict], dim=0)
    skele_masks = torch.stack([dd.pop("skele_masks") for dd in data_dict], dim=0)
    baked = [dd.pop("baked_skeleton") for dd in data_dict]

    if baked[0] is not None:
        baked = torch.stack(baked, dim=0)

    skeletons = [dd.pop("skeletons") for dd in data_dict]

    return images, masks, skeletons, skele_masks, baked


if __name__ == "__main__":
    """
    class dataset(Dataset):
    def __init__(
        self,
        path: Union[List[str], str],
        transforms: Optional[Transform] = lambda x: x,
        pad_size: Optional[int] = 100,
        device: Optional[str] = "cpu",
        sample_per_image: Optional[int] = 1,
    ):
    """
    from skoots.train.merged_transform import merged_transform_3D

    data = dataset(
        path="/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/mitochondria/train/hide",
        transforms=merged_transform_3D,
    )

    for m in data.masks:
        print(m.max(), m.shape)

    for i in range(len(data)):
        print(data[i]["masks"].max(), data[i]["masks"].shape)
