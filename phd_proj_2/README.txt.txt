# Machine Learning Regressor for determining hidden device Parameters from transient J-V curves

This code is an updated and polished version of what was used in the paper "Autoencoder for parameter estimation and current-voltage curve simulation of perovskite solar cells" (https://doi.org/10.1038/s41524-025-01875-0).

## Features
- Autoencoder with condition on the latent space that allows to estimate device Parameters from J-V curves
- After training, the Encoder can be used as a Parameter estimator, the Decoder can be used as a Surrogate device simulator
- K-fold cross validation