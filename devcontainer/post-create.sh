#!/bin/bash
# Post-create script for PSI-HDL GitHub Codespaces
# This script runs automatically after the container is created

set -e

echo "========================================"
echo "PSI-HDL Development Environment Setup"
echo "========================================"

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install main project dependencies
echo "📦 Installing project dependencies..."
pip install -r requirements.txt

# Install additional development tools
echo "🔧 Installing development tools..."
pip install black pylint ipykernel jupyter

# Install PyTorch with CUDA support (if available) or CPU version
echo "🔥 Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "Installing PyTorch (CPU version for Codespaces)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Create output directories if they don't exist
echo "📁 Creating output directories..."
mkdir -p Code/output/burgers
mkdir -p Code/output/laplace
mkdir -p Code/output/snn_xor
mkdir -p Code/output/memristor/figures
mkdir -p Code/output/vteam_comparison
mkdir -p Code/output/cross_validation
mkdir -p Code/output/noise_robustness
mkdir -p Code/output/additional_experiments/epsilon_ablation
mkdir -p Code/output/additional_experiments/larger_snn
mkdir -p Code/output/additional_experiments_2/multi_physics_memristors
mkdir -p Code/output/additional_experiments_2/network_scalability
mkdir -p Code/output/additional_experiments_2/lambda_physics_ablation
mkdir -p Code/output/additional_experiments_2/multiple_seeds
mkdir -p Code/output/additional_experiments_2/baseline_comparison

# Create cache directory for PyTorch
mkdir -p .cache/torch

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name psi-hdl --display-name "PSI-HDL (Python 3.11)"

# Verify installation
echo ""
echo "========================================"
echo "✅ Verifying installation..."
echo "========================================"
python -c "
import numpy as np
import pandas as pd
import torch
import matplotlib
import scipy

print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'Matplotlib version: {matplotlib.__version__}')
print(f'SciPy version: {scipy.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "========================================"
echo "🎉 Setup complete!"
echo "========================================"
echo ""
echo "Quick Start:"
echo "  1. Run Burgers demo:    python Code/demo_psi_hdl.py --model burgers"
echo "  2. Run all demos:       python Code/demo_psi_hdl.py --model all"
echo "  3. Run SNN XOR demo:    python Code/demo_snn_xor.py"
echo "  4. Run Memristor demo:  python Code/demo_memristor.py"
echo "  5. Run all experiments: python Code/run_all_experiments.py"
echo ""
echo "For more information, see README.md"
echo ""