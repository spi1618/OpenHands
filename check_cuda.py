#!/usr/bin/env python3
import os
import subprocess
import sys

print("=== CUDA Environment Check ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Check if nvidia-smi is available
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("nvidia-smi output:")
        print(result.stdout)
    else:
        print("nvidia-smi failed:", result.stderr)
except FileNotFoundError:
    print("nvidia-smi not found")

# Check CUDA version
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("nvcc version:")
        print(result.stdout)
    else:
        print("nvcc failed:", result.stderr)
except FileNotFoundError:
    print("nvcc not found")

# Check PyTorch
try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available in PyTorch")
except ImportError:
    print("PyTorch not available")
except Exception as e:
    print(f"Error checking PyTorch: {e}") 