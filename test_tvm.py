#!/usr/bin/env python3

import sys
import os

print("Testing TVM installation...")

try:
    import tvm
    print(f"✅ TVM imported successfully! Version: {tvm.__version__}")
except ImportError as e:
    print(f"❌ TVM import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch imported successfully! Version: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"✅ TorchVision imported successfully! Version: {torchvision.__version__}")
except ImportError as e:
    print(f"❌ TorchVision import failed: {e}")
    sys.exit(1)

print("✅ All imports successful!")
