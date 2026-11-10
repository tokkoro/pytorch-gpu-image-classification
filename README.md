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
| `valohai.yaml` | Steps and pipeline for running the scripts on [Valohai](https://valohai.com/) |

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

Optional arguments (run any script with `--help` to list them):

```bash
# Choose the device and the output file for training
python step2_train.py --device cpu  --model_path cifar_net_cpu.pt
python step2_train.py --device cuda --model_path cifar_net_gpu.pt

# Change the hyperparameters (defaults are the tutorial's values)
python step2_train.py --epochs 4 --learning_rate 0.001 --momentum 0.9 --batch_size 4

# Evaluate a specific model file
python step3_test.py --model_path cifar_net_gpu.pt
```

The scripts also print their key numbers (loss, accuracy, ...) as JSON lines,
for example `{"accuracy": 53.4, ...}`. Valohai collects these as metadata.

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

## Running on Valohai

[`valohai.yaml`](valohai.yaml) defines three steps, one per script, and a
pipeline:

| Step / pipeline | Script | Inputs | Outputs |
|---|---|---|---|
| `data-preview` | `step1_data.py` | `cifar10` | `train_samples.png` |
| `train` | `step2_train.py` | `cifar10` | `cifar_net.pt` |
| `test` | `step3_test.py` | `cifar10`, `model` | `test_samples.png` |
| `train-and-test` (pipeline) | `train`, then `test` on the trained model | | |

How the code works on Valohai:

- **Docker image:** `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime`, the
  official PyTorch image with CUDA 12.6 (works with NVIDIA drivers that
  support CUDA 12). Each step only adds matplotlib with pip.
- **Data:** the dataset is the `cifar10` input. It defaults to the
  `cifar-10-python.tar.gz` archive on the CIFAR website, which Valohai
  downloads (and caches) to `/valohai/inputs/cifar10/`; the scripts extract
  it and never download data themselves. To use your own copy, upload the
  archive to your project's data store and pass its URL or `datum://` link as
  the `cifar10` input.
- **Parameters:** the command-line options above (`epochs`, `learning_rate`,
  `momentum`, `batch_size`, `num_workers`, `device`) are Valohai parameters,
  passed to the scripts as `--name=value`.
- **Outputs:** files are written to `/valohai/outputs/` (`$VH_OUTPUTS_DIR`), so
  Valohai uploads and versions them. Locally they go to the current directory
  as before.
- **Metadata:** the JSON lines the scripts print become execution metadata:
  `loss` per 2000 mini-batches and `epoch_loss` per epoch during training,
  and `accuracy` plus `accuracy_<class>` for each class from the test step.
  You can chart and compare them in the Valohai UI.

### First-time setup

Install the [Valohai CLI](https://docs.valohai.com/getting-started/quickstart.md),
log in, and create a project in this directory (or link an existing one):

```bash
pip install valohai-cli
vh login
vh project create --name image-recognition-pytorch   # or: vh project link
vh lint                                              # check valohai.yaml
```

To run from Git instead of uploading local code, connect this repository to
the project in the Valohai UI (project Settings > Repository), push your
commits, and run `vh project fetch`.

### Running

```bash
# Run the current local code ("ad hoc", no git push needed) and follow the logs
vh execution run train --adhoc --watch

# Pick a GPU machine (list them with `vh environments`) and change parameters
vh execution run train --adhoc --environment <environment-slug> --epochs=4

# Look at the sample images
vh execution run data-preview --adhoc

# Evaluate a trained model; use the cifar_net.pt URL or datum:// link
# from a train execution's outputs
vh execution run test --adhoc --model=datum://<datum-id>

# Train and test in one go
vh pipeline run train-and-test --adhoc
```

Ad hoc runs upload the working directory with Git packaging, so the
git-ignored `data/` folder and model files aren't uploaded. If no
`--environment` is given, Valohai uses the project's default environment.
