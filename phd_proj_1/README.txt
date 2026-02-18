# Machine Learning Classification for detecting most limiting Parameter

This code is an updated and extended version of what was used in the paper "Identifying Performance Limiting Parameters in Perovskite Solar Cells Using Machine Learning" (https://doi.org/10.1002/solr.202300999).

Comparison of tree-based classifiers (Decision Tree, Random Forest, Extra Trees, XGBoost) for identifying swept simulation parameters from perovskite solar cell device parameters.

## Features
- Multi-class classification of 20 different parameter sweeps
- Comprehensive model comparison with multiple metrics
- Feature importance analysis
- Confusion matrix visualization for each model

## Requirements
- Python 3.8+
- scikit-learn
- xgboost
- pandas, numpy, matplotlib, seaborn