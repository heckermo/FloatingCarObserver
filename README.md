# Floating Car Observers in Intelligent Transportation Systems: Detection Modeling and Temporal Insights

**Code Repository for the Paper Under Review at *Transportation Research Part C: Emerging Technologies***

---

## Overview

This repository, **FloatingCarObservers**, is organized into three main components that reflect the structure of the accompanying paper:

1. **`co-simulation`**  
   - Contains the code for running the co-simulation modeling approach.  
   - Includes scripts for dataset generation used in the emulation approach with neural networks.

2. **`FCO_modeling`**  
   - Implements alternative modeling approaches (2D and 3D ray tracing).  
   - Houses the training scripts for the emulation approach.  
   - Provides code for running these modeling approaches in a SUMO simulation.  
   - Contains scripts to generate datasets for the temporal insights experiments.

3. **`TFCO`**  
   - Offers analysis code for temporal-potential insights.  
   - Defines and trains architectures to recover previously seen vehicles.

Each directory includes its own `README.md` with more detailed instructions and explanations of the corresponding scripts.


## Getting Started

If you only want to work with a subset of this repository (e.g., just `co-simulation` or `TFCO`), you can use **Git Sparse Checkout**:

### 1. Clone Without Checking Out Files

```bash
git clone --no-checkout --filter=blob:none https://github.com/YourUserName/FloatingCarObservers.git
cd FloatingCarObservers
```

### 2. Enable Sparse Checkout

```bash
git sparse-checkout init --cone
```

### 3. Specify Which Directory to Checkout

For `co-simulation` only:

```bash
git sparse-checkout set co-simulation
```

For `FCO_modeling` only:

```bash
git sparse-checkout set FCO_modeling
```

For `TFCO` only:

```bash
git sparse-checkout set TFCO
```

### 4. Checkout the Selected Directories

```bash
git checkout
```

## Setting Up the Conda Environment

Each subcomponent of this repository relies on its own Conda environment due to conflicting dependencies. Follow these steps to set up the environment for each directory:

### 1. Create the Conda Environment

Navigate to the desired directory and create the Conda environment using the provided `environment.yaml` file:

For `co-simulation`:

```bash
cd co-simulation
conda env create -f environment.yaml
```

For `FCO_modeling`:

```bash
cd FCO_modeling
conda env create -f environment.yaml
```

For `TFCO`:

```bash
cd TFCO
conda env create -f environment.yaml
```


## Open Issues
We are continuing to refine this repository while the paper is under review. The following items are actively being addressed:

1. Unified Conda Environment  
   Each subcomponent relies on its own environment due to conflicting dependencies. We plan to merge them into a unified setup.

2. Standardized Configurations and Logging  
   Not all scripts fully utilize .yaml files, and we are also implementing uniform logging for all modules.

3. Loading Trained Emulation Models  
   We will release trained emulation models for various sensor setups and 3D detection algorithms as described in the paper. The optimal distribution method is still being determined to ensure a streamlined, user-friendly experience.

   ## Citation
   If you use this repository in your research, please cite our previous work while the current paper is under review. Additionally, refer to the `README.md` files in the subcomponents of this repository for details on components utilized from other works.

   ```
   @INPROCEEDINGS{10422398,
      author={Gerner, Jeremias and Rößle, Dominik and Cremers, Daniel and Bogenberger, Klaus and Schön, Torsten and Schmidtner, Stefanie},
      booktitle={2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)}, 
      title={Enhancing Realistic Floating Car Observers in Microscopic Traffic Simulation}, 
      year={2023},
      pages={2396-2403},
      doi={10.1109/ITSC57777.2023.10422398}
   }
   ```





