from __future__ import annotations

import numpy as np
import torch

from .dataset import CSI_Dataset, CSI_Ready_Dataset, UT_HAR_dataset, Widar_Dataset
from .models.ntu_fi_model import (
    NTU_Fi_BiLSTM,
    NTU_Fi_CNN_GRU,
    NTU_Fi_GRU,
    NTU_Fi_LSTM,
    NTU_Fi_MLP,
    NTU_Fi_Mamba,
    NTU_Fi_RNN,
    NTU_Fi_ResNet101,
    NTU_Fi_ResNet18,
    NTU_Fi_ResNet50,
    NTU_Fi_LeNet,
    NTU_Fi_ViT,
)
from .models.ut_har_model import (
    UT_HAR_BiLSTM,
    UT_HAR_CNN_GRU,
    UT_HAR_GRU,
    UT_HAR_LSTM,
    UT_HAR_LeNet,
    UT_HAR_MLP,
    UT_HAR_RNN,
    UT_HAR_ResNet101,
    UT_HAR_ResNet18,
    UT_HAR_ResNet50,
    UT_HAR_ViT,
)
from .models.widar_model import (
    Widar_BiLSTM,
    Widar_CNN_GRU,
    Widar_GRU,
    Widar_LSTM,
    Widar_LeNet,
    Widar_MLP,
    Widar_RNN,
    Widar_ResNet101,
    Widar_ResNet18,
    Widar_ResNet50,
    Widar_ViT,
)


def _subset_dataset(dataset, fraction: float, seed: int | None):
    if fraction is None or fraction >= 1.0:
        return dataset
    if fraction <= 0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    size = max(1, int(len(dataset) * fraction))
    rng = np.random.default_rng(0 if seed is None else seed)
    indices = rng.permutation(len(dataset))[:size].tolist()
    return torch.utils.data.Subset(dataset, indices)


def _ntu_num_classes(dataset_name: str) -> int:
    return {"NTU-Fi-HumanID": 14, "NTU-Fi_HAR": 6, "APPLIED": 3}[dataset_name]


def load_data_n_model(
    dataset_name: str,
    model_name: str,
    root: str,
    *,
    seq_len: int = 500,
    pooling: str = "mean",
    mamba_selective: str = "on",
    shuffle_subcarriers: bool = False,
    shuffle_antennas: bool = False,
    shuffle_seed: int = 0,
    train_fraction: float = 1.0,
    noise: str = "none",
    noise_p: float = 0.0,
    val_noise: str = "none",
    val_noise_p: float | None = None,
    seed: int | None = None,
    batch_train: int = 64,
    batch_test: int = 64,
    num_workers: int = 0,
    model_params: dict | None = None,
):
    model_params = model_params or {}
    val_noise_p = noise_p if val_noise_p is None else val_noise_p

    classes = {"UT_HAR_data": 7, "NTU-Fi-HumanID": 14, "NTU-Fi_HAR": 6, "Widar": 22, "APPLIED": 3}

    if dataset_name == "UT_HAR_data":
        data = UT_HAR_dataset(root)
        train_set = torch.utils.data.TensorDataset(data["X_train"], data["y_train"])
        test_set = torch.utils.data.TensorDataset(
            torch.cat((data["X_val"], data["X_test"]), 0),
            torch.cat((data["y_val"], data["y_test"]), 0),
        )
        train_set = _subset_dataset(train_set, train_fraction, seed)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(batch_train),
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=max(1, int(batch_test)),
            shuffle=False,
            num_workers=int(num_workers),
        )
        if model_name == "MLP":
            model = UT_HAR_MLP()
            train_epoch = 200
        elif model_name == "LeNet":
            model = UT_HAR_LeNet()
            train_epoch = 200
        elif model_name == "ResNet18":
            model = UT_HAR_ResNet18()
            train_epoch = 200
        elif model_name == "ResNet50":
            model = UT_HAR_ResNet50()
            train_epoch = 200
        elif model_name == "ResNet101":
            model = UT_HAR_ResNet101()
            train_epoch = 200
        elif model_name == "RNN":
            model = UT_HAR_RNN()
            train_epoch = 3000
        elif model_name == "GRU":
            model = UT_HAR_GRU()
            train_epoch = 200
        elif model_name == "LSTM":
            model = UT_HAR_LSTM()
            train_epoch = 200
        elif model_name == "BiLSTM":
            model = UT_HAR_BiLSTM()
            train_epoch = 200
        elif model_name == "CNN+GRU":
            model = UT_HAR_CNN_GRU()
            train_epoch = 200
        elif model_name == "ViT":
            model = UT_HAR_ViT()
            train_epoch = 200
        else:
            raise ValueError(f"Unsupported model for UT_HAR_data: {model_name}")
        return train_loader, test_loader, model, train_epoch

    if dataset_name == "Widar":
        train_set = Widar_Dataset(root + "Widardata/train/")
        test_set = Widar_Dataset(root + "Widardata/test/")
        train_set = _subset_dataset(train_set, train_fraction, seed)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=int(batch_train),
            shuffle=True,
            num_workers=int(num_workers),
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=max(1, int(batch_test)),
            shuffle=False,
            num_workers=int(num_workers),
        )
        if model_name == "MLP":
            model = Widar_MLP(classes["Widar"])
            train_epoch = 30
        elif model_name == "LeNet":
            model = Widar_LeNet(classes["Widar"])
            train_epoch = 100
        elif model_name == "ResNet18":
            model = Widar_ResNet18(classes["Widar"])
            train_epoch = 100
        elif model_name == "ResNet50":
            model = Widar_ResNet50(classes["Widar"])
            train_epoch = 100
        elif model_name == "ResNet101":
            model = Widar_ResNet101(classes["Widar"])
            train_epoch = 100
        elif model_name == "RNN":
            model = Widar_RNN(classes["Widar"])
            train_epoch = 500
        elif model_name == "GRU":
            model = Widar_GRU(classes["Widar"])
            train_epoch = 200
        elif model_name == "LSTM":
            model = Widar_LSTM(classes["Widar"])
            train_epoch = 200
        elif model_name == "BiLSTM":
            model = Widar_BiLSTM(classes["Widar"])
            train_epoch = 200
        elif model_name == "CNN+GRU":
            model = Widar_CNN_GRU(classes["Widar"])
            train_epoch = 200
        elif model_name == "ViT":
            model = Widar_ViT(num_classes=classes["Widar"])
            train_epoch = 200
        else:
            raise ValueError(f"Unsupported model for Widar: {model_name}")
        return train_loader, test_loader, model, train_epoch

    if dataset_name not in ("NTU-Fi-HumanID", "NTU-Fi_HAR", "APPLIED"):
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_classes = _ntu_num_classes(dataset_name)
    if dataset_name == "APPLIED":
        train_set = CSI_Ready_Dataset(
            root + "APPLIED/train_amp/",
            seq_len=seq_len,
            shuffle_subcarriers=shuffle_subcarriers,
            shuffle_antennas=shuffle_antennas,
            shuffle_seed=shuffle_seed,
            noise=noise,
            noise_p=noise_p,
        )
        test_set = CSI_Ready_Dataset(
            root + "APPLIED/test_amp/",
            seq_len=seq_len,
            shuffle_subcarriers=shuffle_subcarriers,
            shuffle_antennas=shuffle_antennas,
            shuffle_seed=shuffle_seed,
            noise=val_noise,
            noise_p=val_noise_p,
        )
        train_epoch_default = {
            "MLP": 50,
            "LeNet": 50,
            "ResNet18": 50,
            "ResNet50": 60,
            "ResNet101": 60,
            "RNN": 80,
            "GRU": 50,
            "LSTM": 50,
            "BiLSTM": 50,
            "CNN+GRU": 60,
            "ViT": 50,
            "Mamba": 60,
        }
    elif dataset_name == "NTU-Fi_HAR":
        train_set = CSI_Dataset(
            root + "NTU-Fi_HAR/train_amp/",
            seq_len=seq_len,
            shuffle_subcarriers=shuffle_subcarriers,
            shuffle_antennas=shuffle_antennas,
            shuffle_seed=shuffle_seed,
            noise=noise,
            noise_p=noise_p,
        )
        test_set = CSI_Dataset(
            root + "NTU-Fi_HAR/test_amp/",
            seq_len=seq_len,
            shuffle_subcarriers=shuffle_subcarriers,
            shuffle_antennas=shuffle_antennas,
            shuffle_seed=shuffle_seed,
            noise=val_noise,
            noise_p=val_noise_p,
        )
        train_epoch_default = {
            "MLP": 30,
            "LeNet": 30,
            "ResNet18": 30,
            "ResNet50": 30,
            "ResNet101": 30,
            "RNN": 70,
            "GRU": 30,
            "LSTM": 30,
            "BiLSTM": 30,
            "CNN+GRU": 100,
            "ViT": 30,
            "Mamba": 60,
        }
    else:  # NTU-Fi-HumanID
        # SenseFi protocol uses test_amp for training.
        train_set = CSI_Dataset(
            root + "NTU-Fi-HumanID/test_amp/",
            seq_len=seq_len,
            shuffle_subcarriers=shuffle_subcarriers,
            shuffle_antennas=shuffle_antennas,
            shuffle_seed=shuffle_seed,
            noise=noise,
            noise_p=noise_p,
        )
        test_set = CSI_Dataset(
            root + "NTU-Fi-HumanID/train_amp/",
            seq_len=seq_len,
            shuffle_subcarriers=shuffle_subcarriers,
            shuffle_antennas=shuffle_antennas,
            shuffle_seed=shuffle_seed,
            noise=val_noise,
            noise_p=val_noise_p,
        )
        train_epoch_default = {
            "MLP": 50,
            "LeNet": 50,
            "ResNet18": 50,
            "ResNet50": 50,
            "ResNet101": 50,
            "RNN": 75,
            "GRU": 50,
            "LSTM": 50,
            "BiLSTM": 50,
            "CNN+GRU": 200,
            "ViT": 50,
            "Mamba": 75,
        }

    train_set = _subset_dataset(train_set, train_fraction, seed)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=int(batch_train),
        shuffle=True,
        num_workers=int(num_workers),
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=int(batch_test),
        shuffle=False,
        num_workers=int(num_workers),
    )

    if model_name == "MLP":
        model = NTU_Fi_MLP(num_classes)
    elif model_name == "LeNet":
        model = NTU_Fi_LeNet(num_classes)
    elif model_name == "ResNet18":
        model = NTU_Fi_ResNet18(num_classes)
    elif model_name == "ResNet50":
        model = NTU_Fi_ResNet50(num_classes)
    elif model_name == "ResNet101":
        model = NTU_Fi_ResNet101(num_classes)
    elif model_name == "RNN":
        model = NTU_Fi_RNN(num_classes)
    elif model_name == "GRU":
        model = NTU_Fi_GRU(num_classes)
    elif model_name == "LSTM":
        model = NTU_Fi_LSTM(num_classes)
    elif model_name == "BiLSTM":
        model = NTU_Fi_BiLSTM(num_classes)
    elif model_name == "CNN+GRU":
        model = NTU_Fi_CNN_GRU(num_classes)
    elif model_name == "ViT":
        vit_cfg = dict(model_params.get("vit", {}))
        model = NTU_Fi_ViT(num_classes=num_classes, **vit_cfg)
    elif model_name == "Mamba":
        mamba_cfg = dict(model_params.get("mamba", {}))
        model = NTU_Fi_Mamba(
            num_classes,
            pooling=pooling,
            selective=(mamba_selective != "off"),
            **mamba_cfg,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    train_epoch = int(train_epoch_default[model_name])
    return train_loader, test_loader, model, train_epoch
