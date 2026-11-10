"""Shared pieces of the CIFAR10 tutorial (plan.md): data loading, network, helpers."""
import json
import os
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI windows; images are saved to PNG instead

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2

MODEL_FILENAME = 'cifar_net.pt'
PATH = './' + MODEL_FILENAME
batch_size = 4
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Name of the Valohai input that holds the CIFAR10 archive (see valohai.yaml)
DATASET_INPUT = 'cifar10'

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def on_valohai():
    """Valohai sets VH_INPUTS_DIR / VH_OUTPUTS_DIR in every execution."""
    return 'VH_INPUTS_DIR' in os.environ


def input_dir(name):
    """Directory of a Valohai input (/valohai/inputs/<name>), or None when running locally."""
    if not on_valohai():
        return None
    return Path(os.environ['VH_INPUTS_DIR']) / name


def outputs_dir():
    """Valohai uploads everything written here; locally, files go to the current directory."""
    return os.getenv('VH_OUTPUTS_DIR', '.')


def output_path(filename):
    return os.path.join(outputs_dir(), filename)


def log_metadata(**values):
    """Print one JSON line; Valohai picks it up as execution metadata (metrics)."""
    print(json.dumps(values), flush=True)


def prepare_data(root='./data'):
    """Return (root, download) to pass to torchvision.datasets.CIFAR10.

    Locally the dataset is downloaded into ./data as in the tutorial. On Valohai
    it comes from the `cifar10` input: either the extracted
    `cifar-10-batches-py` directory or the `cifar-10-python.tar.gz` archive,
    which is extracted into `root` here.
    """
    data_input = input_dir(DATASET_INPUT)
    if data_input is None:
        return root, True
    if (data_input / 'cifar-10-batches-py').is_dir():
        return str(data_input), False
    archives = sorted(data_input.glob('**/*.tar.gz'))
    if not archives:
        raise FileNotFoundError(f'No CIFAR10 archive (*.tar.gz) found in Valohai input {data_input}')
    print(f'Extracting {archives[0]} to {root}')
    with tarfile.open(archives[0]) as tar:
        tar.extractall(root, filter='data')
    return root, False


def get_loaders(batch_size=batch_size, num_workers=2):
    root, download = prepare_data()
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def imshow(img, filename):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f'Saved image grid to {filename}')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
