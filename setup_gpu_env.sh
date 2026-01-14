#!/bin/bash
# Setup GPU environment variables for Isaac Sim
# This script should be sourced in the Docker container

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export EGL_VISIBLE_DEVICES=${EGL_VISIBLE_DEVICES:-0}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}
export QT_X11_NO_MITSHM=${QT_X11_NO_MITSHM:-1}
export MESA_GL_VERSION_OVERRIDE=${MESA_GL_VERSION_OVERRIDE:-4.5}
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export PRIVACY_CONSENT=${PRIVACY_CONSENT:-Y}

echo "=== GPU Environment Variables Set ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "EGL_VISIBLE_DEVICES: $EGL_VISIBLE_DEVICES"
echo "VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
echo "VK_DRIVER_FILES: $VK_DRIVER_FILES"
echo "DISPLAY: ${DISPLAY:-NOT SET}"








