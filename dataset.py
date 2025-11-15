import glob
import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

CSI_DEFAULT_MEAN = float(os.getenv("NTU_FI_NORM_MEAN", "42.3199"))
CSI_DEFAULT_STD = float(os.getenv("NTU_FI_NORM_STD", "4.9802"))

def set_csi_normalization(mean: float, std: float) -> None:
    """Override module-level CSI normalization parameters at runtime.

    This updates the values used by CSI_Dataset.__getitem__ for normalization.
    """
    global CSI_DEFAULT_MEAN, CSI_DEFAULT_STD
    CSI_DEFAULT_MEAN = float(mean)
    CSI_DEFAULT_STD = float(max(std, 1e-8))


def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize (overridable via env vars for different dataset releases)
        x = (x - CSI_DEFAULT_MEAN)/CSI_DEFAULT_STD
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x,y


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')
        
        # normalize
        x = (x - 0.0025)/0.0119
        
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y


class CSI_Ready_Dataset(Dataset):
    """Dataset for pre-shaped CSI amplitude tensors.

    Expects MATLAB files with key 'CSIamp' shaped as (S, T), (1, S, T) or (3, S, T)
    where S=114 and T=500. No downsampling or reshaping is applied. Optionally,
    a single stream can be tiled to three to match existing model interfaces.
    """

    def __init__(self, root_dir: str, tile_to_three: bool = True):
        self.root_dir = root_dir
        self.tile_to_three = tile_to_three
        self.data_list = glob.glob(root_dir + '/*/*.mat')
        self.folder = glob.glob(root_dir + '/*/')
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        mat = sio.loadmat(sample_dir)
        if 'CSIamp' not in mat:
            raise KeyError(f"CSIamp not found in {sample_dir}")
        x = mat['CSIamp']
        # Normalize
        x = (x - CSI_DEFAULT_MEAN) / CSI_DEFAULT_STD
        # Ensure 3D shape (streams, 114, 500)
        x = np.array(x)
        if x.ndim == 2:
            # (S, T) -> (1, S, T)
            x = x[None, ...]
        if x.shape[-2:] != (114, 500):
            # try to transpose if common alternative (500,114)
            if x.shape[-2:] == (500, 114):
                x = np.transpose(x, (0, 2, 1))
            else:
                raise ValueError(f"Unexpected CSIamp shape {x.shape} in {sample_dir}")
        if self.tile_to_three and x.shape[0] == 1:
            x = np.repeat(x, 3, axis=0)
        x = torch.FloatTensor(x)
        return x, y
