"Build a robust image classification pipeline for scientific image data with noisy crowd-sourced labels"

In this project, I want to classify the morphology of galaxies based on images.


This project uses the Galaxy Zoo dataset available on Kaggle:

https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge

Due to Kaggle’s data usage policies, the dataset is not included in this repository.
Please download it directly from Kaggle.


Since computation capacities are limited, the models (there are several approaches i want to test in the future) are not directly compared against each other. it should just show some possible ways one could go.

gm_model.py:
	using the pre-trained resnet18, which is the smallest renet (https://doi.org/10.48550/arXiv.1512.03385). useful for fast experimentation and when 	hardware capabilities are limited. Further description of the architecture can be found for example here: 	https://www.kaggle.com/code/iamtapendu/introduction-to-resnet-18