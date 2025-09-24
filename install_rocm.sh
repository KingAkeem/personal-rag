#!/bin/bash

echo "Installing ROCm for AMD GPU support..."

# Update system
sudo apt update

# Install ROCm (version 6.0)
wget https://repo.radeon.com/amdgpu-install/6.0.0/ubuntu/focal/amdgpu-install_6.0.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.0.60000-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm --no-dkms

# Add your user to the render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Verify installation
echo "Verifying ROCm installation..."
/opt/rocm/bin/rocminfo

echo "ROCm installation complete. Please reboot your system."