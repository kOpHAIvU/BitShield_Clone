# BitShield - Windows Setup Guide

## Overview

This is the research artifact for the paper [BitShield: Defending Against
Bit-Flip Attacks on DNN Executables](https://www.ndss-symposium.org/ndss-paper/bitshield-defending-against-bit-flip-attacks-on-dnn-executables/)
in NDSS 2025, adapted for Windows.

## Prerequisites

### Required Software

1. **Python 3.8** - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation
   - Verify installation: `python --version`

2. **Git** - Download from [git-scm.com](https://git-scm.com/download/win)
   - Verify installation: `git --version`

3. **Docker Desktop** - Download from [docker.com](https://www.docker.com/products/docker-desktop/)
   - Required for building models and running experiments
   - Make sure Docker Desktop is running before using Docker commands

### Optional Software

4. **ImageNet Dataset** (Optional)
   - If you have ImageNet dataset, set the environment variable:
   ```cmd
   set IMAGENET_ROOT=C:\path\to\your\imagenet\dataset
   ```

## Getting Started

### 1. Initial Setup

Run the Windows setup script:

```cmd
setup.bat
```

This script will:
- Initialize git submodules
- Create a Python virtual environment
- Install all required dependencies
- Create necessary directories

### 2. Activate Environment

Before working on the project, always activate the environment:

```cmd
env.bat
```

Or manually:
```cmd
venv\Scripts\activate.bat
```

### 3. Download Datasets

Download required datasets:

```cmd
python tools\ensure_datasets.py
```

This will download:
- CIFAR10, CIFAR100, MNIST, FashionMNIST
- DTD, GTSRB datasets
- LPIPS weights
- FID Inception weights

### 4. Docker Setup (Required for Model Building)

Build the Docker image for model compilation:

```cmd
docker\setup.bat
```

This will:
- Build a Docker image with all required dependencies
- Install Ghidra for binary analysis
- Build TVM and Glow compilers

## Usage

### Training Models

Train models using the training script:

```cmd
REM Train ResNet50 on CIFAR10
python support\models\train.py resnet50 CIFAR10

REM Train multiple models
for %m in (resnet50 densenet121 googlenet) do (
    for %x in (CIFAR10 MNISTC FashionC) do (
        python support\models\train.py %m %x
    )
)
```

### Building Models

After training, build the models:

```cmd
REM Build all models defined in cfg.py
docker\run-in-docker.bat python buildmodels.py

REM Or use DVC to reproduce the build pipeline
dvc repro
```

### Running Experiments

#### Bit-Flip Sweep

Find vulnerable bits in models:

```cmd
REM Sweep for vulnerable bits in ResNet50 on CIFAR10
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10
```

#### Attack Simulation

Run attack simulations:

```cmd
REM Run all attack simulations
tools\runattacksim.bat

REM Run specific model/dataset combination
tools\runattacksim.bat -m resnet50 -d CIFAR10
```

### Using DVC

This project uses DVC for data version control:

```cmd
REM Pull latest data
dvc pull

REM Reproduce experiments
dvc repro

REM Check status
dvc status
```

## File Structure

### Windows-Specific Files

- `setup.bat` - Initial setup script
- `env.bat` - Environment activation script
- `docker\setup.bat` - Docker image setup
- `docker\run-in-docker.bat` - Run commands in Docker
- `docker\config.bat` - Docker configuration
- `tools\runattacksim.bat` - Attack simulation runner

### Key Directories

- `datasets/` - Downloaded datasets
- `models/` - Trained model weights
- `built/` - Compiled model binaries
- `results/` - Experiment results
- `ghidra/` - Binary analysis files
- `compilers/` - TVM, Glow, NNFusion compilers

## Troubleshooting

### Common Issues

1. **Python not found**
   - Make sure Python 3.8 is installed and in PATH
   - Try: `python --version`

2. **Docker not running**
   - Start Docker Desktop
   - Check: `docker version`

3. **Permission errors**
   - Run Command Prompt as Administrator
   - Check file permissions

4. **Git submodule issues**
   - Run: `git submodule update --init --recursive`

5. **Virtual environment issues**
   - Delete `venv/` directory and run `setup.bat` again

### Performance Tips

1. **Use SSD storage** for better I/O performance
2. **Allocate more memory** to Docker Desktop (8GB+ recommended)
3. **Use WSL2 backend** for Docker (better performance)

### Memory Requirements

- **Minimum**: 8GB RAM
- **Recommended**: 16GB+ RAM
- **Storage**: 50GB+ free space

## Development

### Adding New Models

1. Add model definition in `support/models/`
2. Update `cfg.py` with new model configurations
3. Train the model using `support/models/train.py`
4. Build using `buildmodels.py`

### Adding New Datasets

1. Add dataset to `tools/ensure_datasets.py`
2. Update `cfg.py` with dataset configurations
3. Update `utils.py` if needed for dataset-specific properties

## Support

For issues specific to the Windows setup:
1. Check this README first
2. Verify all prerequisites are installed
3. Check Docker Desktop is running
4. Ensure sufficient disk space and memory

For general project issues, refer to the main README.md file. 