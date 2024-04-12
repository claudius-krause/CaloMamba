# CaloMamba 
# Graph Mamba Model for Pileup Subtraction

## Overview
This Graph Mamba model is designed for pileup subtraction. It processes the constituent particle information from the Zj process (Z > νν).

## Model Input
The input file contains data on constituent particles, and the average pileup number per bunch crossing is set to 60. Consequently, each event features approximately 1500-4000 particles.

## Model Objective
The primary aim of the Graph Mamba model is to determine the hard energy fraction (\(\hat{y} = \frac{E_{LV}}{E_{LV} + E_{PU}}\)) of each Energy Flow object (EFlowPhoton, EFlowNeutralHadron, and EFlowChargedHadron) through a regression task.

## Data Segmentation
Both the training and validation datasets are divided into five parts:

1. **Features**: Includes \( p_T \), eta, phi, E, particle ID, and vertex ID for each particle in every event.
2. **Mask**: A binary mask where 1 indicates particles with \( p_T > 0 \) and 0 represents particles with \( p_T = 0 \) (currently not applied).
3. **R1 Matrix**: Collects the \Delta_\eta and \Delta_\phi information for all charged PU particles within a radius of 0.3 around each particle, serving as input for the GNN network.
4. **R0 Matrix**: Gathers \Delta_\eta and \Delta_\phi information for all charged LV particles within a radius of 0.3 around each particle, also for GNN input.
5. **Rm1 Matrix**: Assembles \Delta_\eta and \Delta_\phi data for all neutral particles within a radius of 0.3 around each particle, used in the GNN network.

## Running the Model
Execute the model by running:
```bash
python GraphMamba.py

## Dataset Samples
A small sample containing 10 events is available for both the training and validation datasets.

## Additional Resources
Check the input details and model settings in the `pileup_subtraction_note.pdf`. Updates will be made to the README file in the forthcoming days.

## Model Tuning
The model configuration has not been fine-tuned. Please feel free to play with it!.

