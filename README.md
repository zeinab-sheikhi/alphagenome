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
- **Purpose**: Processes sequences using multi-head attention with pairwise interaction features
- **Architecture**: 9 transformer layers with alternating sequence and pairwise updates
- **Multi-Query Attention**: 8 query heads sharing single key/value heads for efficiency
- **Key Features**:
  - RoPE (Rotary Position Embedding) for positional encoding up to 8,192 positions
  - RMS batch normalization for stable training
  - Soft-clipped attention logits in range [-5, 5] using tanh activation
  - Attention bias injection from pairwise features
  - Residual connections with MLP blocks (2x expansion ratio)
  - Dropout regularization (default 0.1)
- **Dimensions**: Operates on sequences of length 8,192 with configurable feature dimensions

### 3. Pairwise Interaction Block
- **Purpose**: Models long-range interactions between sequence positions using pairwise representations
- **Architecture**: Three-stage processing pipeline
- **Components**:
  - **SequenceToPairBlock**: Converts sequence features to pairwise representations
    - Average pooling from 8,192 → 512 positions for computational efficiency
    - Multi-head relative position encoding with learnable biases
    - Central mask features for position-aware interactions
  - **RowAttentionBlock**: Self-attention within pairwise dimension
    - Per-row attention across pairwise features
    - RMS normalization with projection layers
  - **PairMLPBlock**: Feed-forward processing of pairwise features
    - 2x channel expansion with ReLU activation
    - Dropout regularization
- **Output**: Pairwise attention bias matrices fed back to transformer layers

### 4. Sequence Decoder
- **Purpose**: Reconstructs full-resolution sequences from transformer output using U-Net architecture
- **Process**: 7-stage upsampling from 128bp back to 1bp resolution
- **Architecture**: Symmetric to encoder with residual connections
- **Key Features**:
  - **UpResBlock**: Residual upsampling blocks with skip connections
  - Learnable residual scaling (default 0.9 initialization)
  - 2x upsampling at each stage using repeat_interleave
  - Skip connections from corresponding encoder stages
  - Progressive channel reduction: 1536 → 768 channels
  - 1x1 convolutions for channel alignment in skip connections
- **Output**: Full-resolution sequence features ready for task-specific heads

### 5. Task-Specific Output Heads
- **Purpose**: Convert decoded sequence features to task-specific predictions
- **Implementation**: Configurable output layers (not implemented in base model)
- **Typical Tasks**:
  - Gene expression prediction
  - Chromatin accessibility
  - Protein binding site identification
  - Variant effect prediction
  - Epigenetic mark prediction
- **Architecture**: Usually simple linear projections or small MLPs
- **Output**: Task-dependent (e.g., regression values, classification logits)

