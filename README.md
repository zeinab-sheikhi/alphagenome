# AlphaGenome

A PyTorch implementation of the AlphaGenome architecture for genomic sequence modeling, based on the paper ["AlphaGenome: A Foundation Model for Genomic Sequence Modeling"](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v2.full.pdf).

## Overview

AlphaGenome is a foundation model designed for genomic sequence modeling that combines convolutional neural networks with transformer architectures to process and understand DNA sequences at multiple resolutions. The model can handle sequences up to 1 million base pairs (1Mbp) and is trained on diverse genomic tasks.

## Architecture

The AlphaGenome architecture consists of 5 main components:

### 1. Sequence Encoder
- **Purpose**: Progressively downsamples input DNA sequences from 1bp resolution to 128bp resolution
- **Process**: 7-stage downsampling with increasing channel depth (768 → 1536 channels)
- **Output**: Multi-resolution feature representations for U-net skip connections
- **Key Features**:
  - RMS batch normalization
  - Standardized 1D convolutions with weight standardization
  - Residual connections with zero-padding for channel dimension changes
  - Max pooling with stride 2 for downsampling

### 2. Transformer Tower

### 3. Pairwise Interaction Block

### 4. Sequence Decoder

### 5. Task-Specific Output Heads

