"""Import neccessary packages"""
import numpy as np
import matplotlib
from matplotlib import pyplot
import json
import pandas as pd
from sklearn import ensemble, linear_model, metrics
import pickle

import atools
from atools_ml.dataio import df_setup
from atools_ml.prep import dimensionality_reduction


"""From the binned data"""
def bin_df(descriptors_df, n_bins, target):
    """
    This function can be used to split a data frame into
    'n_bins' different entries in a dictionary. The n_bins
    parameter specifies the number of bins to split the file into.

    Parameters
    ----------
    descriptors_df: pandas.DataFrame
        the data to split into n_bins bins
    n_bins: int
        Number of desired bins
    target: str
        Target columns we want to split by


    Returns
    ---------
    results: dict

    Note: will save out n_bins number of csv files
    """
    msg='{} is not a field in the given data file.'.format(target)
    assert (target in descriptors_df.columns), msg

    results = dict()
    for i in range(n_bins):
        bin_min = descriptors_df[target].quantile(i/float(n_bins))
        bin_max = descriptors_df[target].quantile((i+1)/float(n_bins))
        df_temp = descriptors_df[(descriptors_df[target] >= bin_min)
                                    & (descriptors_df[target] <= bin_max)]
        results['{}_{}'.format(target, i)] = df_temp
    return results


def split_df(descriptors_df,
             test_fraction,
             training_fractions,
             n_bins, target,
             output_dir,
             overwrite=False):
    """ From the a descriptors dataframe, save out n_sets evenly distributed df

    Parameters
    ----------
    descriptors_df: pandas.DataFrame
        the data to be splitted
    test_fraction : floatd
        Fraction of the input data set that is used exclusively
        for testing. This portion will be removed to create
        a training set, which will be used for to create
        different training set with different fractions
    training_fractions : list of float
        Fractions of the training data set that is going to be saved out to.
        For example [0.1, 0.3, 0.5, 0.8, 1] will save out 5 files
        with 10%, 30%, 50%, 80% and 100% of the grand data set, respectively.
    target : str
        Target columns that we want to split (evenly distribution criteria)
    output_dir : str
        Directory where all the splitted csv is going to be saved out to
    overwrite : bool, optional, default=False
        Option to whether or not overwrite the csv files
    """
    binned_target = bin_df(descriptors_df,
                           n_bins,
                           target)
    binned_training = binned_target
    target_test = pd.DataFrame()
    target_trainings = dict()

    # First create the testing set and drop those columns from
    # the binned_training set
    for n in range(n_bins):
        test_tmp = binned_target['{}_{}'.format(target, n)].sample(frac=test_fraction)
        target_test = target_test.append(test_tmp)
        binned_training['{}_{}'.format(target, n)] = binned_training['{}_{}'.format(target, n)].drop(test_tmp.index)

    # Need overwrite option check (use os.path.isfile)
    # Raise warning (or print something out) and do nothing
    # Saving out the testing csv
    target_test.to_csv('{}/{}_test.csv'.format(output_dir, target))

    # Then create the target_training set
    final_target = dict()
    for fraction in training_fractions:
        final_target['{}_{}'.format(target, fraction)] = pd.DataFrame()
        for n in range(n_bins):
            final_target['{}_{}'.format(target, fraction)] = final_target['{}_{}'.format(target, fraction)].append(binned_training['{}_{}'.format(target, n)].sample(frac=fraction))
        # Need overwrite option check (use os.path.isfile)
        # Raise warning (or print something out) and do nothing
        # Saving out the training csv
        filename = '{}/{}_{}.csv'.format(output_dir, target, fraction)

        if overwrite:
            write_csv = True
        else:
            import os
            if not os.path.isfile(filename):
                write_csv = True
            else:
                write_csv = False

        if write_csv:
            final_target['{}_{}'.format(target, fraction)].to_csv(filename)
        else:
            continue

    return {'test_set': target_test,
            'train_set': final_target}


def train_rf(data, target, output_path,
             overwrite=False, seed=43):
    """ Train and save the machine learning models

    Parameters
    ----------
    data : str or DataFrame (or list of str or DataFrame)
        Can pass in either a str (or a list of str) as path to
        the data file (in csv format), or a DataFrame or
        a list of DataFrame
    target : str
        'COF' or 'intercept'
    output_path : str
        Path to file where regressor model will be pickled to
        (Notice, no extension needed, the model will be automatically saved
        to .pickle format and the features list will be saved to a .txt format
        with the same name)
    seed : int
        Random seed for the machine learning model

    Returns
    -------
    model : sklearn.pipeline.Pipeline
    """
    # Load the data
    if isinstance(data, (str, pd.DataFrame)):
        data = [data]

    # Can also consider raise an error if the inpur is not in the correct format

    identifiers = ['terminal_group_1',
                   'terminal_group_2',
                   'terminal_group_3',
                   'backbone',
                   'frac-1',
                   'frac-2']
    targets= ['COF', 'intercept']

    loaded_df = pd.DataFrame()
    for item in data:
        if isinstance(data, str):
            tmp_df = pd.read_csv(item, index_col=0)
        elif isinstance(data, pd.DataFrame):
            tmp_df = data
        loaded_df = loaded_df.append(data, ignore_index=True)

    to_drop = ['pc+-mean', 'pc+-diff',
               'pc+-min', 'pc+-max',
               'pc--mean', 'pc--diff',
               'pc--min', 'pc--max']

    # Reduce the number of features by running them thorugh the original
    # dimensionality_reduction by Andrew,
    # Will used the new dimensionality reduction in later version
    features = list(loaded_df.drop(targets + identifiers, axis=1))
    # Change this to only pick the top 3 features
    # (may need to hand pick these from the feature importances plots)

    if target == 'COF':
        top_3 = ['hk-alpha-mean', 'pbf-mean', 'asphericity-mean']
    elif target == 'intercept':
        top_3 = ['tpsa-min', 'hbonds', 'vsa-fpos-min']

    X_train, y_train = loaded_df[top_3], loaded_df[target]
    regr = ensemble.RandomForestRegressor(
                            n_estimators=1000,
                            oob_score=True,
                            random_state=seed)
    regr.fit(X_train, y_train)

    if overwrite:
        write_model = True
    else:
        import os
        if not os.path.isfile(output_path+'.pickle'):
            write_model = True
        else:
            write_model = False
            
    if write_model:
        with open(output_path+'.pickle', 'wb') as mfile:
            print('Model saved out to {}'.format(output_path+'.pickle'))
            pickle.dump(regr, mfile)
        with open(output_path+'.ptxt', 'wb') as lfile:
            print('Features saved out to {}'.format(output_path+'.ptxt'))
            pickle.dump(top_3, lfile)

    return {'model': regr,
            'features': top_3}



if __name__ == '__main__':
    overwrite = False
    """ Split the 50-50 mixed df and the everything df"""
    mixed5050 = pd.read_csv('../../data/raw-data/skimmed-mixed-50-50.csv', index_col=0)
    splitted5050_path = '../../data/splitted-data/mixed5050'
    everything = pd.read_csv('../../data/raw-data/everything.csv', index_col=0)
    splitted_everything_path = '../../data/splitted-data/everything'


    splitted5050 = {'COF': {'train_set': dict(), 'test_set': None},
                    'intercept': {'train_set': dict(), 'test_set': None}}
    splitted_everything = {'COF': {'train_set': dict(), 'test_set': None},
                           'intercept': {'train_set': dict(), 'test_set': None}}

    """ Only need to run this block once to create splitted data set
        Can just load data from file the next time

    for target in ['COF', 'intercept']:
        splitted5050[target] = split_df(descriptors_df=mixed5050,
                                test_fraction=0.2,
                                training_fractions=[0.05, 0.1, 0.2,
                                                   0.3, 0.5, 0.7, 1],
                                n_bins=6,
                                target=target,
                                output_dir=splitted5050_path,
                                overwrite=overwrite)

        splitted_everything[target] = split_df(descriptors_df=everything,
                                test_fraction=0.2,
                                training_fractions=[0.01, 0.02, 0.03, 0.05, 0.1,
                                                   0.2, 0.3, 0.5, 0.7, 1],
                                n_bins=6,
                                target=target,
                                output_dir=splitted_everything_path,
                                overwrite=overwrite)
    """
    mfractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
    efractions = [0.01, 0.02, 0.03, 0.05, 0.1,
                  0.2, 0.3, 0.5, 0.7, 1]
    for target in ['COF', 'intercept']:
        for fraction in mfractions:
            model_name = '{}_{}'.format(target, fraction)
            filename = '{}/{}.csv'.format(splitted5050_path,
                                          model_name)
            splitted5050[target]['train_set'][model_name] = pd.read_csv(filename, index_col=0)
        tfilename = '{}/test_df.csv'.format(splitted5050_path)
        splitted5050[target]['test_set'] = pd.read_csv(tfilename, index_col=0)

    for target in ['COF', 'intercept']:
        for fraction in efractions:
            model_name = '{}_{}'.format(target, fraction)
            filename = '{}/{}.csv'.format(splitted_everything_path,
                                          model_name)
            splitted_everything[target]['train_set'][model_name] = pd.read_csv(filename, index_col=0)
        tfilename = '{}/test_df.csv'.format(splitted5050_path)
        splitted_everything[target]['test_set'] = pd.read_csv(tfilename, index_col=0)

    """ Train the models"""
    original_csv = pd.read_csv('../../data/raw-data/original-100.csv', index_col=0)
    original_models = dict()
    mixed5050_models = dict()
    everything_models = dict()
    for target in ['COF', 'intercept']:
        # Train the original models
        omodel_path = '../specialModels/original/{}'.format(target)
        original_models[target] = train_rf(data=original_csv,
                     target=target,
                     output_path=omodel_path,
                     overwrite=overwrite)
        # Train the mixed5050 models
        for model in splitted5050[target]['train_set']:
            m5model_path = '../specialModels/mixed5050/{}'.format(model)
            mixed5050_models[model] = train_rf(data=splitted5050[target]['train_set'][model],
                         target=target,
                         output_path=m5model_path,
                         overwrite=overwrite)
        # Train the everything model
        for model in splitted_everything[target]['train_set']:

            emodel_path = '../specialModels/everything/{}'.format(model)
            everything_models[model] = train_rf(
                        data=splitted_everything[target]['train_set'][model],
                        target=target,
                        output_path=emodel_path,
                        overwrite=overwrite)

