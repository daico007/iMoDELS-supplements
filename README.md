# iMoDELS supplemental information

This repository is intended to host the supplemental information for the paper title: "High-throughput Screening of Tribological Properties of Monolayer Films using Molecular Dynamics and Machine Learning"

### MD simulations data

The tribological data calculated from MD simulations can be found as `csv` files located [here](data/raw-data). 
These properties are calculated using the Amonton's Law of Friction. The friction force of normal load is collected from the last 5 ns of the MD production run. Please refer to the main paper for more details. 



### Machine Learning (Random Forest Regressor)

This study utilized the Random Forest Regressor provided by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). 
Multiple ML models were created, each varied by the number data points of their training set and/or the seed used to select data points out of the main data set. 



### Using this repository

To set up the environment neccessary for this repo 
```
    conda env create -f env.yml
    conda activate screening
```
This repository includes codes needed to:
    1. Train and pickle the  ML models: [trainML.py](random_forest/src/trainML.py)
    2. Evaluate the created ML models by applying them to common test set: [predictML.py](random_forest/src/predictML.py)
    3. Analyze/visualize the final result: series of iPython notebook located [here](random_forest/src/)
    4. Once the pickled ML models have been created, the [Data-Lookup.ipynb](Data-Lookup.ipynb) can be used to look up data from the main data set, or utilized the created ML models to estimate tribological properties of any systems.
