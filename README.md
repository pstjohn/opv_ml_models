# Scripts and optimized model weights for predicting material optoelectronic properties

These files are intended to accompany an upcoming manuscript on accurately predicting optoelectronic properties of large molecules without 3D coordinates: [Message-passing neural networks for high-throughput polymer screening](https://arxiv.org/abs/1807.10363)

Models use the [NREL/nfp](https://github.com/NREL/nfp) library to provide the additional needed Keras layers.

The `data/` directory contains a link to download the OPV database, either with or without optimized 3D coordinates.

Scripts to train the different models are located in the top-level directory:
* `run_schnet_b3lyp.py`: To train a model on 3D coordinates, following the [SchNet with Edge-updates model structure](https://arxiv.org/abs/1806.03146)
* `run_2d_model_mae.py`: To train a model to predict all eight targets at once from only SMILES strings
* `run_2d_model_single_target.py`: To train a model to predict only of the prediction targets.

The single-target models are invoked with a command line argument (integer) that specifies which target should be fit.

Trained models can be evaluated following the syntax shown in `evaluate_models_mae.ipynb`. Creation of the train and test databases follows the outline presented in `database_processing_and_duplicate_detection.ipynb`.
