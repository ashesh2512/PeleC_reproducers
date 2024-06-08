#!/bin/sh

load_nvidia_env
nvcc -std=c++17 -g -lineinfo -Xptxas=-v -DNUM_SPECIES=53 -DNUM_STEPS=7 -o CKUBMS CKUBMS_wrapper.cu