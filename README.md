# ConsistentLES-ANN
Integrated DNS-LES solver for use in training numerically consistent artificial neural network closure models. One program outputs resolved SGS quantities which are then used as inputs and training data for an ANN specified in another code.

This ANN predicts the SGS stress tensor term $\tau_{ij}$ from invariants in the output LES data. These Galilean invariant terms form the basis of a Galilean invariant SGS data-driven model. Previous work by Beck et al. (2019) also attempts "perfect" data driven modelling of the SGS stress tensor. However, we aim to improve the physicality of the model obtained by her group by ensuring Galilean invariance.
