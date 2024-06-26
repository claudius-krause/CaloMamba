# Graph Mamba: Pileup Subtraction and Exploration in CaloMamba

## Overview
This Graph Mamba model is designed for pileup subtraction. It processes the constituent particle information from the Zj process (Z > νν).

## Model Input
The input file contains data on constituent particles for each Zj(Z > v v) event, and the average pileup number per bunch crossing is set to 60. Consequently, each event features approximately 1500-4000 particles.

## Model Objective
The primary aim of the Graph Mamba model is to determine the hard energy fraction $\hat{y} = \frac{E_{LV}}{E_{LV} + E_{PU}}$ of each Energy Flow object (EFlowPhoton, EFlowNeutralHadron, and EFlowChargedHadron) through a regression task. After that, each EFlow Object's energy can be rescaled and then passed to Delphes for jet clustering and object reconstruction. In this way, we can achieve the pileup subtraction with GraphMamba model.

## Dataset Samples
A small sample containing 10 events and a medium sample containing 100 events are available for both the training and validation datasets.

## Additional Resources
Check the input details and Mamba model settings in the `pileup_subtraction_note.pdf`.

## Model Tuning
The model configuration has not been fine-tuned. Please feel free to play with it!.

## Data Segmentation
Both the training and validation datasets are divided into five parts:

1. **Features**: Includes $p_T$, $\eta$, $\phi$, E, particle ID, and vertex ID for each particle in every event.
2. **Mask**: A binary mask where 1 indicates particles with $p_T > 0$ and 0 represents particles with $p_T = 0$ (currently not applied).
3. **R1 Matrix**: Collects the $\Delta \eta$ and $\Delta \phi$ information for all charged PU particles within a radius of 0.3 around each particle, serving as input for the GNN network.
4. **R0 Matrix**: Gathers $\Delta \eta$ and $\Delta \phi$ information for all charged LV particles within a radius of 0.3 around each particle, also for GNN input.
5. **Rm1 Matrix**: Assembles $\Delta \eta$ and $\Delta \phi$ data for all neutral particles within a radius of 0.3 around each particle, used in the GNN network.

## Running the Model
Execute the model by running:
```bash
python GraphMamba.py



