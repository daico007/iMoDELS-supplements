"""Import neccessary packages"""
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn import ensemble, linear_model, metrics, model_selection
import pickle
import os

def input_prep(system, features, descriptors=None):
    """Preparing the SMILES strings for prediction

    Parameters
    ----------
    system : dict
        A description of the system in form of a dictionary.
        The dictionary should be in the form
        system = {'bottom' : [('H-SMILES1', 'CH3-SMILES1', frac1),
                            ('H-SMILES2', 'CH3-SMILES2', frac2)]
                  'top' : [('H-SMILES3', 'CH3-SMILES3', frac3),
                         ('H-SMILES4', 'CH3-SMILES4', frac4]
        (Try to make this works with n-components for each
        monolayer)
    features : list or str
        List of features used for prediction (model-specific)
        or the path to the features.
    descriptors : df or str or None, optional, default=None
        DataFrame of all the independent descriptors
        (terminal group specific) or path to the csv file.
        If None is given, we will determine this on the flight, which
        may take a bit longer

    Returns
    -------
    output : pd.DataFrame
         The DataFrame that can be input into the random forest
         model for prediction
    """
    if isinstance(features, str):
        with open(features, 'rb') as f:
            features = pickle.load(f)
    if isinstance(descriptors, str):
        with open(descriptors, 'r'):
            descriptors = pd.read_csv(descriptors, index_col=0)
    # Turn a df into a dict, key is SMILES str (both h-SMILES and ch3-SMIELS)
    descriptors_dict = descriptors.to_dict()

    # Calculate average of bottom
    desc_top = dict()
    for idx, tg in enumearte(system['bottom']):
        desc_top['tg_{}'.format(idx)] = {
            'h': descriptors_dict.get(tg[0]),
            'ch3' : descriptors_dict.get(tg[1]),
            'frac' : descriptors_dict.get(tg[2])}

    # Calculate average of top
    for idx, tg in enumearte(system['top']):
        desc_top['tg_{}'.format(idx)] = {
            'h': descriptors_dict.get(tg[0]),
            'ch3' : descriptors_dict.get(tg[1]),
            'frac' : descriptors_dict.get(tg[2])}

    return None

def predict(system, model, features, descriptors):
    """Load the model and predict

    Parameters
    ----------
    system : dict
        A description of the system in form of a dictionary.
        The dictionary should be in the form
        system = {bottom : [(H-SMILES1, CH3-SMILES1, frac1),
                            (H-SMILES2, CH3-SMILES2, frac2)]
                  top : [(H-SMILES3, CH3-SMILES3, frac3),
                         (H-SMILES4, CH3-SMILES4, frac4]
    model : sklearn.ensemble.RandomForestRegressor or str
        The random forest model or the path to the random forest
        model used to predict
    features : list or str
        The list of features presented in the model (ordered)
        or the path to load in the features
    descriptors : df or str
        DataFrame with all the independent descriptors
        (SMILES string specific) or path to the csv file

    Returns
    -------
    results : dict
        Dictionary of COF and F0 result
    """
    if isinstance(model, str):
        with open(model, 'rb') as f:
            model = pickle.load(f)
    if isinstance(features, str):
        with open(features, 'rb') as f:
            features = pickle.load(f)

    assert isinstance(model, sklearn.ensemble.RandomForestRegressor)
    assert len(features) == model.n_features_

    return None

def evaluate_model(test_df, model, target, features, descriptors, output_path):
    """Calculate the r-square of the model when apply on the test data frame

    Paramters
    ---------
    test_df : pd.DataFrame or str
        Test DataFrame or path to the test DataFrame (csv format)
    model : sklearn.ensemble.RandomForestRegressor or str
        The random forest model or the path to the random forest
        model that needed to be tested
    target : str
        The target that
    features : list or str
        The list of features presented in the model (ordered)
        or the path to load in the features
    descriptors : df or str
        DataFrame with all the independent descriptors
        (SMILES string specific) or path to the csv file.
        Only needed in certain case (TBD), should be set to
        None for now.
    output_path : str
        Path to save out the result (as a json file)

    Returns
    -------
    result : dict()
    """
    import os
    from pathlib import Path
    if isinstance(test_df, str):
        test_df = pd.read_csv(test_df)
    if isinstance(model, str):
        with open(model, 'rb') as f:
            model = pickle.load(f)
    if isinstance(features, str):
        with open(features, 'rb') as f:
            fearures = pickle.load(f)

    assert isinstance(model, sklearn.ensemble.RandomForestRegressor)
    assert len(features) == model.n_features_

    test_df = test_df.sort_index(axis=0)
    test_df.reset_index(inplace=True)

    results = dict()
    results[target] = dict()
    test_df_red = test_df.filter(features + [target], axis=1)

    for idx, row in test_df_red.iterrows():
        simulated = row.pop(target)
        predicted = model.predict(np.asarray(row).reshape(1,-1))
        results[target][idx] = {
            'tg-1': test_df.iloc[idx]['terminal_group_1'],
            'frac-1': test_df.iloc[idx]['frac-1'],
            'tg-2': test_df.iloc[idx]['terminal_group_2'],
            'frac-2': test_df.iloc[idx]['frac-2'],
            'tg-3': test_df.iloc[idx]['terminal_group_3'],
            'predicted-{}'.format(target): predicted[0],
            'simulated-{}'.format(target): simulated}

    # Calculate r square
    test_df_X = test_df_red.filter(features, axis=1)
    test_df_Y = test_df_red.filter([target], axis=1)
    results[target]['r_square'] = model.score(test_df_X, test_df_Y)
    print('Saving out to {}.'.format(output_path))

    opath = Path(output_path)
    if not os.path.isdir(opath.parent):
        os.mkdir(opath.parent)
    with open(output_path, 'w') as f:
        json.dump(results, f)

    return results

if __name__ == '__main__':
    # Define paths for data, model, test, and output
    for bins in [10]:
        for i in range(5):
            for path in [f'../predicted-results/original/nbins-{bins}',
                         f'../predicted-results/mixed5050/nbins-{bins}',
                         f'../predicted-results/everything/nbins-{bins}']:
                if not os.path.isdir(path):
                    os.mkdir(path)
            oresults_path = f'../predicted-results/original/nbins-{bins}'
            mresults_path = f'../predicted-results/mixed5050/nbins-{bins}/set_{i}'
            eresults_path = f'../predicted-results/everything/nbins-{bins}/set_{i}'

            omodels_path = f'../models/original/'
            mmodels_path = f'../models/mixed5050/nbins-{bins}/set_{i}'
            emodels_path = f'../models/everything/nbins-{bins}/set_{i}'

            models_results = {omodels_path : oresults_path,
                            mmodels_path : mresults_path,
                            emodels_path : eresults_path
                            }

            test_5050_path = f'../../data/splitted-data/mixed5050/nbins-{bins}/test_set.csv'
            test_2575_path = f'../../data/splitted-data/mixed2575/nbins-{bins}/test_set.csv'
            test_everything_path = f'../../data/splitted-data/everything/nbins-{bins}/test_set.csv'

            test_5050 = pd.read_csv(test_5050_path, index_col=0)
            test_2575 = pd.read_csv(test_2575_path, index_col=0)
            test_everything = pd.read_csv(test_everything_path, index_col=0)

            tests = {'5050' : test_5050,
                     '2575' : test_2575,
                     'everything' : test_everything
                    }

            for path in models_results:
                for entry in os.scandir(path):
                    # Ouput dir is models specific (original, 5050, everything)
                    output_path = models_results[path]
                    if entry.name.endswith('pickle'):
                        if 'intercept' in entry.name:
                            target = 'intercept'
                        elif 'COF' in entry.name:
                            target = 'COF'

                        name = os.path.splitext(entry.name)[0]
                        features_path = f'{path}/{name}.ptxt'

                        # Load model and features
                        with open(entry.path, 'rb') as fmodel:
                            model = pickle.load(fmodel)
                        with open(features_path, 'rb') as ffeatures:
                            features = pickle.load(ffeatures)

                        # Throw everything to the evaluation method
                        for test_name, test_set in tests.items():
                            output = f'{output_path}/{name}_on_{test_name}.json'
                            evaluate_model(test_df=test_set,
                                           model=model,
                                           target=target,
                                           features=features,
                                           descriptors=None,
                                           output_path=output)
