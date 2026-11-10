# CIFAR10 Image Classifier (PyTorch)

A small convolutional neural network that classifies 32×32 colour images into
10 classes (plane, car, bird, cat, deer, dog, frog, horse, ship, truck).

The code follows the official PyTorch tutorial
[Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
(the original tutorial source is in [`plan.md`](plan.md)), split into separate
scripts so each stage can be run on its own.

## Project layout

| File | What it does |
|---|---|
| `common.py` | Shared code: data transforms and loaders, the `Net` model, image-saving helper |
| `step1_data.py` | Tutorial step 1: downloads CIFAR10 and saves sample training images |
| `step2_train.py` | Tutorial steps 2–4: builds the network, trains for 2 passes over the data, saves the model |
| `step3_test.py` | Tutorial step 5: evaluates a saved model on the 10,000 test images |
| `plan.md` | The original tutorial |

## Setup

Requires Python 3.10+ and, optionally, an NVIDIA GPU with a recent driver.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pip install torch` installs the CUDA build by default on Linux. For other
platforms or a specific CUDA version, see the
[PyTorch install selector](https://pytorch.org/get-started/locally/).

Check that PyTorch sees your GPU (optional):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Getting the data

The dataset is **not** stored in this repository (`data/` is git-ignored).
It's downloaded automatically into `./data` the first time any script runs:

```bash
python step1_data.py
```

That fetches `cifar-10-python.tar.gz` (about 170 MB) from
<https://www.cs.toronto.edu/~kriz/cifar.html> and extracts it to
`data/cifar-10-batches-py/`.

To download it by hand instead (for example on a machine without internet
access for Python):

```bash
mkdir -p data
curl -L -o data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf data/cifar-10-python.tar.gz -C data
```

torchvision checks the archive's checksum and skips the download if the files
are already there.

## Usage

Run the scripts from the project root, in order:

```bash
# 1. Download data and save 4 sample training images to train_samples.png
python step1_data.py

# 2-4. Train (uses the GPU if available) and save the model to cifar_net.pt
python step2_train.py

# 5. Evaluate cifar_net.pt on the test set; saves test_samples.png
python step3_test.py
```

Optional arguments:

```bash
# Choose the device and the output file for training
python step2_train.py cpu  cifar_net_cpu.pt
python step2_train.py cuda cifar_net_gpu.pt

# Evaluate a specific model file
python step3_test.py cifar_net_gpu.pt
```

Trained models (`*.pt`) and the generated PNG images are git-ignored. Rerun
the scripts to recreate them.

## Expected results

Two passes of training take about 40 s on an RTX 4080 Laptop GPU, or a bit
over a minute on a multi-core CPU. The final training loss is around 1.3, and
accuracy on the test set is **about 53–54%**; random guessing would get 10%.
Per-class accuracy varies noticeably between runs. Classes like plane, frog
and truck tend to score highest, and cat, dog and bird lowest.

The GPU gives only a modest speedup here because the network is tiny and the
batch size is 4. Widening the network (the exercise at the end of the
tutorial) makes the difference larger.
