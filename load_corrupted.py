import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GEMCorruptedDataset(Dataset):
    """
    GEM Corrupted Dataset loader for robustness evaluation.
    Supports MNIST-C and CIFAR-10-C.
    """

    def __init__(self, root_dir, noise_type, dataset_type="MNIST", split='test',
                 labels_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.noise_type = noise_type
        self.dataset_type = dataset_type
        self.split = split
        self.labels_file = labels_file

        # File paths
        if dataset_type == "MNIST":
            images_file = os.path.join(root_dir, "mnist_c", noise_type, f"{split}_images.npy")
            labels_file = os.path.join(root_dir, "mnist_c", noise_type, f"{split}_labels.npy")
        elif dataset_type == "CIFAR-10":
            images_file = os.path.join(root_dir, "CIFAR-10-C", f"{noise_type}.npy")
            if labels_file is None:
                raise ValueError("labels_file must be specified for CIFAR-10 dataset.")
            labels_file = os.path.join(root_dir, "CIFAR-10-C", labels_file)
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        # Check files
        if not os.path.exists(images_file):
            raise FileNotFoundError(f"Corrupted images file not found: {images_file}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Corrupted labels file not found: {labels_file}")

        # Load data
        self.images = np.load(images_file).astype(np.float32) / 255.0
        self.labels = np.load(labels_file)

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Mismatch: images ({len(self.images)}) vs labels ({len(self.labels)})"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        # Convert to Tensor
        image = torch.from_numpy(image).float()

        if self.dataset_type == "MNIST":
            # Case 1: if (28,28) -> add channel
            if image.ndim == 2:
                image = image.unsqueeze(0)  # (1,28,28)

            # Case 2: if (28,28,1) -> transpose axes
            elif image.ndim == 3 and image.shape[-1] == 1:
                image = image.permute(2, 0, 1)  # (1,28,28)

        elif self.dataset_type == "CIFAR-10":
            # CIFAR-10 → (H,W,C) → (C,H,W)
            if image.ndim == 3:
                image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_gem_corrupted_data(data_dir, noise_type, dataset_type="MNIST",
                               split='test', labels_file=None, batch_size=64,
                               num_workers=2):
    """
    Prepare corrupted data loader with appropriate transformations.
    """
    if dataset_type == "MNIST":
        transform = transforms.Normalize((0.1307,), (0.3081,))
    elif dataset_type == "CIFAR-10":
        transform = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    dataset = GEMCorruptedDataset(
        data_dir, noise_type,
        dataset_type=dataset_type,
        split=split,
        labels_file=labels_file,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return dataloader


if __name__ == "__main__":
    # Test on MNIST-C
    data_dir_mnist = "./data"
    noise_type_mnist = "brightness"
    try:
        test_loader_mnist = prepare_gem_corrupted_data(
            data_dir_mnist, noise_type_mnist,
            dataset_type="MNIST", split='test', batch_size=64
        )
        images, labels = next(iter(test_loader_mnist))
        print(f"MNIST-C: {images.shape}, {labels.shape}")
    except Exception as e:
        print(f"Error MNIST-C: {e}")

    # Test on CIFAR-10-C
    data_dir_cifar = "./data"
    noise_type_cifar = "brightness"
    try:
        test_loader_cifar = prepare_gem_corrupted_data(
            data_dir_cifar, noise_type_cifar,
            dataset_type="CIFAR-10", labels_file="labels.npy", batch_size=64
        )
        images, labels = next(iter(test_loader_cifar))
        print(f"CIFAR-10-C: {images.shape}, {labels.shape}")
    except Exception as e:
        print(f"Error CIFAR-10-C: {e}")


# ===== helpers for corruption loaders (append-only, standalone) =====
try:
    import os as _os
    import numpy as _np
    import torch as _torch
    from torch.utils.data import TensorDataset as _TensorDataset, DataLoader as _DataLoader
    from torchvision import transforms as _transforms
except Exception:
    pass

def make_cifar10c_loader_fn(data_dir="./data", num_workers=4):
    """Return get_loader(corruption: str, severity: int, batch_size: int) -> DataLoader"""
    def get_loader(corruption: str, severity: int, batch_size: int):
        images_root = _os.path.join(data_dir, "CIFAR-10-C")
        labels_path = _os.path.join(images_root, "labels.npy")
        if not _os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing CIFAR-10-C labels: {labels_path}")
        labels_all = _np.load(labels_path).astype(_np.int64)  # (50_000,)

        img_path = _os.path.join(images_root, f"{corruption}.npy")
        if not _os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        imgs_all = _np.load(img_path).astype(_np.float32)  # (50_000, 32, 32, 3)

        if not (1 <= int(severity) <= 5):
            raise ValueError("severity must be in {1..5}")
        n_per = imgs_all.shape[0] // 5
        s0 = (int(severity) - 1) * n_per
        s1 = int(severity) * n_per

        imgs = imgs_all[s0:s1] / 255.0
        labs = labels_all[s0:s1]

        X = _torch.from_numpy(imgs).permute(0,3,1,2).contiguous()
        y = _torch.from_numpy(labs)

        norm = _transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        X = norm(X)

        ds = _TensorDataset(X, y)
        return _DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=_torch.cuda.is_available())
    return get_loader

def make_mnistc_loader_fn(data_dir="./data", num_workers=4, with_severity_dirs=False):
    """Return get_loader(corruption: str, severity: int, batch_size: int) -> DataLoader"""
    def _load_pair(img_path, lab_path):
        if not (_os.path.exists(img_path) and _os.path.exists(lab_path)):
            raise FileNotFoundError(f"Missing MNIST-C: {img_path} / {lab_path}")
        X = _np.load(img_path).astype(_np.float32)  # (N,28,28) or (N,28,28,1)
        y = _np.load(lab_path).astype(_np.int64)
        if X.ndim == 3:
            X = X[:, None, :, :]
        elif X.ndim == 4 and X.shape[-1] == 1:
            X = _np.transpose(X, (0,3,1,2))
        X /= 255.0
        X = _torch.from_numpy(X)
        y = _torch.from_numpy(y)
        norm = _transforms.Normalize((0.1307,), (0.3081,))
        X = norm(X)
        return X, y

    def get_loader(corruption: str, severity: int, batch_size: int):
        root = _os.path.join(data_dir, "mnist_c")
        if with_severity_dirs:
            img_path = _os.path.join(root, corruption, str(int(severity)), "test_images.npy")
            lab_path = _os.path.join(root, corruption, str(int(severity)), "test_labels.npy")
        else:
            if int(severity) != 1:
                # for compatibility with loops over severities, hide higher levels
                raise FileNotFoundError("MNIST-C without severity dirs: only severity=1 is available")
            img_path = _os.path.join(root, corruption, "test_images.npy")
            lab_path = _os.path.join(root, corruption, "test_labels.npy")

        X, y = _load_pair(img_path, lab_path)
        ds = _TensorDataset(X, y)
        return _DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=_torch.cuda.is_available())
    return get_loader
