This is a Graph Mamba model for pileup subtraction.

The input file is the constituent particles information of Zj process (Z > v v).

After setting the average pileup number per bunch crossing to 60, each event contains around 1500-4000 particles.

The objective of the GraphMamba model is to obtain the hard energy fraction of each Energy Flow object (EFlowPhoton, EFlowNeutralHadron and EFlowChargedHadron) \hat{y}=\frac{E_{LV}}{E_{LV}+E_{PU}} through a regression task. 

Both the train dataset and the validation dataset are divided by 5 parts:

1, features: p_T, eta, phi, E, particle ID and vertex ID of each particle in the each event.
2, mask: 1 for particles with pt>0 and 0 for particles with pt = 0 (not applied currently).
3, R1 matrix: Collecting the delta_eta and delta_phi information of all the charged PU particles within a radius of 0.3 around each particle as input to the GNN network.
4, R0 matrix: Collecting the delta_eta and delta_phi information of all the charged LV particles within a radius of 0.3 around each particle as input to the GNN network.
5, Rm1 matrix: Collecting the delta_eta and delta_phi information of all the neutral particles within a radius of 0.3 around each particle as input to the GNN network.



Just run python GraphMamba.py to train the model.

There is one small sample containing 10 events for both train and validation dataset.

You can check the input information and Mamba model setting in the pileup_subtraction_note.pdf. I will keep updating the readme file in the next few days.

The model file has not been fine tuned. Please feel free to play with the model!



