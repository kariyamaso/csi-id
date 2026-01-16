import glob
import hashlib
import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

CSI_DEFAULT_MEAN = float(os.getenv("NTU_FI_NORM_MEAN", "42.3199"))
CSI_DEFAULT_STD = float(os.getenv("NTU_FI_NORM_STD", "4.9802"))


def set_csi_normalization(mean: float, std: float) -> None:
    global CSI_DEFAULT_MEAN, CSI_DEFAULT_STD
    CSI_DEFAULT_MEAN = float(mean)
    CSI_DEFAULT_STD = float(max(std, 1e-8))


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _make_perm(size: int, seed: int) -> np.ndarray:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return torch.randperm(size, generator=gen).cpu().numpy()


def _apply_noise(x: np.ndarray, noise: str, noise_p: float) -> np.ndarray:
    if noise == "none" or noise_p <= 0:
        return x
    if noise == "time_mask":
        seq_len = x.shape[-1]
        mask_len = max(1, int(seq_len * noise_p))
        start = np.random.randint(0, max(1, seq_len - mask_len + 1))
        x[..., start : start + mask_len] = 0
        return x
    if noise == "subcarrier_dropout":
        num_sub = x.shape[-2]
        drop = np.random.rand(num_sub) < noise_p
        if drop.any():
            x[:, drop, :] = 0
        return x
    if noise == "gaussian":
        x = x + np.random.normal(0.0, noise_p, size=x.shape)
        return x
    return x


def UT_HAR_dataset(root_dir: str):
    data_list = glob.glob(root_dir + "/UT_HAR/data/*.csv")
    label_list = glob.glob(root_dir + "/UT_HAR/label/*.csv")
    wifi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split("/")[-1].split(".")[0]
        with open(data_dir, "rb") as f:
            data = np.load(f)
            data = data.reshape(len(data), 1, 250, 90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        wifi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split("/")[-1].split(".")[0]
        with open(label_dir, "rb") as f:
            label = np.load(f)
        wifi_data[label_name] = torch.Tensor(label)
    return wifi_data


class Widar_Dataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir + "/*/*.csv")
        self.folder = glob.glob(root_dir + "/*/")
        self.category = {self.folder[i].split("/")[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split("/")[-2]]
        x = np.genfromtxt(sample_dir, delimiter=",")
        x = (x - 0.0025) / 0.0119
        x = x.reshape(22, 20, 20)
        x = torch.FloatTensor(x)
        return x, y


class CSI_Dataset(Dataset):
    """CSI dataset (MAT files under root_dir/*/*.mat)."""

    def __init__(
        self,
        root_dir: str,
        modal: str = "CSIamp",
        transform=None,
        seq_len: int = 500,
        shuffle_subcarriers: bool = False,
        shuffle_antennas: bool = False,
        shuffle_seed: int = 0,
        noise: str = "none",
        noise_p: float = 0.0,
    ):
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform
        self.seq_len = int(seq_len)
        self.shuffle_subcarriers = bool(shuffle_subcarriers)
        self.shuffle_antennas = bool(shuffle_antennas)
        self.shuffle_seed = int(shuffle_seed)
        self.noise = str(noise)
        self.noise_p = float(noise_p)
        self.data_list = glob.glob(root_dir + "/*/*.mat")
        self.folder = glob.glob(root_dir + "/*/")
        self.category = {self.folder[i].split("/")[-2]: i for i in range(len(self.folder))}
        base_seed = self.shuffle_seed + _stable_hash(root_dir)
        self._subcarrier_perm = _make_perm(114, base_seed + 17) if self.shuffle_subcarriers else None
        self._antenna_perm = _make_perm(3, base_seed + 31) if self.shuffle_antennas else None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split("/")[-2]]
        x = sio.loadmat(sample_dir)[self.modal]

        x = (x - CSI_DEFAULT_MEAN) / CSI_DEFAULT_STD
        x = np.array(x)
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        if x.shape[0] != 3 * 114:
            raise ValueError(f"Unexpected CSI shape {x.shape} in {sample_dir}")
        orig_len = x.shape[1]
        if orig_len < self.seq_len:
            raise ValueError(f"seq_len={self.seq_len} exceeds available length {orig_len} in {sample_dir}")
        if orig_len != self.seq_len:
            step = max(1, orig_len // self.seq_len)
            x = x[:, ::step]
            if x.shape[1] < self.seq_len:
                raise ValueError(f"Downsampled length {x.shape[1]} < seq_len={self.seq_len} in {sample_dir}")
            x = x[:, : self.seq_len]
        x = x.reshape(3, 114, self.seq_len)

        if self._antenna_perm is not None:
            x = x[self._antenna_perm, :, :]
        if self._subcarrier_perm is not None:
            x = x[:, self._subcarrier_perm, :]

        if self.transform:
            x = self.transform(x)
        x = _apply_noise(x, self.noise, self.noise_p)
        x = torch.FloatTensor(x)
        return x, y


class CSI_Ready_Dataset(Dataset):
    """Dataset for pre-shaped CSI amplitude tensors stored in mat files.

    Expects 'CSIamp' shaped as (S,T), (1,S,T), or (3,S,T) with S=114.
    """

    def __init__(
        self,
        root_dir: str,
        tile_to_three: bool = True,
        seq_len: int = 500,
        shuffle_subcarriers: bool = False,
        shuffle_antennas: bool = False,
        shuffle_seed: int = 0,
        noise: str = "none",
        noise_p: float = 0.0,
    ):
        self.root_dir = root_dir
        self.tile_to_three = tile_to_three
        self.seq_len = int(seq_len)
        self.shuffle_subcarriers = bool(shuffle_subcarriers)
        self.shuffle_antennas = bool(shuffle_antennas)
        self.shuffle_seed = int(shuffle_seed)
        self.noise = str(noise)
        self.noise_p = float(noise_p)
        self.data_list = glob.glob(root_dir + "/*/*.mat")
        self.folder = glob.glob(root_dir + "/*/")
        self.category = {self.folder[i].split("/")[-2]: i for i in range(len(self.folder))}
        base_seed = self.shuffle_seed + _stable_hash(root_dir)
        self._subcarrier_perm = _make_perm(114, base_seed + 17) if self.shuffle_subcarriers else None
        self._antenna_perm = _make_perm(3, base_seed + 31) if self.shuffle_antennas else None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split("/")[-2]]
        mat = sio.loadmat(sample_dir)
        if "CSIamp" not in mat:
            raise KeyError(f"CSIamp not found in {sample_dir}")
        x = mat["CSIamp"]
        x = (x - CSI_DEFAULT_MEAN) / CSI_DEFAULT_STD

        x = np.array(x)
        if x.ndim == 2:
            x = x[None, ...]
        if x.shape[-2:] != (114, 500):
            if x.shape[-2:] == (500, 114):
                x = np.transpose(x, (0, 2, 1))
            else:
                raise ValueError(f"Unexpected CSIamp shape {x.shape} in {sample_dir}")
        if self.tile_to_three and x.shape[0] == 1:
            x = np.repeat(x, 3, axis=0)

        time_len = x.shape[-1]
        if time_len < self.seq_len:
            raise ValueError(f"seq_len={self.seq_len} exceeds available length {time_len} in {sample_dir}")
        if time_len != self.seq_len:
            step = max(1, time_len // self.seq_len)
            x = x[..., ::step]
            if x.shape[-1] < self.seq_len:
                raise ValueError(f"Downsampled length {x.shape[-1]} < seq_len={self.seq_len} in {sample_dir}")
            x = x[..., : self.seq_len]

        if self._antenna_perm is not None:
            x = x[self._antenna_perm, :, :]
        if self._subcarrier_perm is not None:
            x = x[:, self._subcarrier_perm, :]

        x = _apply_noise(x, self.noise, self.noise_p)
        return torch.FloatTensor(x), y
