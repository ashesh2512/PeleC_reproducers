#!/bin/sh

load_nvidia_env
nvcc -std=c++17 -g -O2 -lineinfo -Xptxas=-v -DNUM_SPECIES=53 -o fKernelSpec fKernelSpec.cu