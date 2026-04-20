# Stable-Baselines3 1.8 requires gym==0.21; its setup.py fails under pip's default build isolation + setuptools>=65.
# This script pins setuptools, installs gym with --no-build-isolation, then installs the rest.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

pip install -U pip
pip install "setuptools>=59.5,<65.0" wheel
pip install "gym==0.21.0" --no-build-isolation
# CPU-only PyTorch from PyTorch index (default pip torch on Windows is often CUDA and can fail with WinError 1114 on c10.dll)
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
