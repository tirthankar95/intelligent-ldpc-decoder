# RL-Layered LDPC Decoder (Old Implementation)

This directory contains an older implementation of the Reinforcement Learning-based Layered LDPC Decoder. This approach uses RL to find optimal sequences of layers for faster and more efficient LDPC decoding.

## Overview

Low-Density Parity-Check (LDPC) codes are powerful error-correcting codes used in modern communication systems. Traditional layered LDPC decoding processes layers in a fixed order, but this implementation uses reinforcement learning to dynamically determine the optimal layer processing sequence.

## Key Components

### Core Modules

- **Global.py** - Contains global variables and parameters used throughout the project
  - `Layers`: Number of decoding layers (default: 9)
  - `subGroups`: Size of layer subgroups (default: 3)
  - `n`: Codeword length (default: 15)
  - `k`: Message length (default: 6)

- **Graph.py** - Creates the state transition graph for the reinforcement learning environment

- **LDPCmain.py** - Main LDPC decoder implementation with reinforcement learning integration
  - Implements layered LDPC decoding algorithm
  - Contains RL reward calculation functions
  - Handles iterative decoding process

- **LDPCHelper.py** - Helper functions for LDPC operations
  - MIN function for minimum calculations
  - Other utility functions for decoding

## Algorithm Details

### Layer Grouping
If L represents the number of layers, the algorithm calculates:
```
L! + L!/1! + L!/2! + ... + L!/(L-1)!
```

For example, with 9 layers (0,1,2,...,8):
- Taking 3 layers at a time: 9C3 combinations
- Size of each layer group = 9/3 = 3

### Decoding Process
1. Initialize parity check matrix H
2. Split H into layer matrices
3. Perform iterative layered decoding
4. Use RL to optimize layer processing order

## Usage

1. Set parameters in `Global.py`
2. Run the main decoding function from `LDPCmain.py`
3. Monitor convergence and performance metrics

## Output Files

- **op.txt** - Contains decoding results and performance logs

## Notes

This is an older implementation. For the current version, see the parent directory (`../rl-layered-ldpc/`).

## Future Improvements

- Store state values in files for persistence
- Group codes based on their response to different layering strategies
- Implement more sophisticated reward functions
