"""Import neccessary packages"""
import os
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
             n_bins,
             target,
             output_dir,
             test_fraction=None,
             test_points=None,
             opt_in_test=None,
             training_fractions=None,
             training_points=None,
             n_tset=1,
             overwrite=False,
             predefined_test=None,
             seed=-1):
    """ From the a descriptors dataframe, save out n_sets evenly distributed df

    Parameters
    ----------
    descriptors_df: pandas.DataFrame
        the data to be splitted
    n_bins : int
        Number of bins to split the data into
    target : str
        Target columns that we want to split (evenly distribution criteria)
    output_dir : str
        Directory where all the splitted csv is going to be saved out to
    test_fraction : float or None
        Fraction of the input data set that is used exclusively
        for testing. This portion will be removed to create
        a training set, which will be used for to create
        different training set with different fractions
    test_points : int or None
        Number of data points that we want to use and create the testing set,
        must be less than the provided data set.
        Can be used in place of the test_fraction variable (one or the other)
    opt_in_test : pd.DataFrametest , optional, default=None
        Dataframe with the optimal systems, everything matched by index.
        Hence, the top DF need be created from the same csv and preserve the
        idx.
    training_fractions : list of float or None
        Fractions of the training data set that is going to be saved out to.
        For example [0.1, 0.3, 0.5, 0.8, 1] will save out 5 files
        with 10%, 30%, 50%, 80% and 100% of the grand data set, respectively.
    training_points : list of int or None
        Number of training data of each data set, must be less than the provided
        data set (after carving out the testing set)
        Can be used in place of the test_fraction variable (one or the other).
        Note: if the maximum of training points is greater that the provided
        data set (after carving out the testing set), this method will create a
        training set with everything.
    overwrite : bool, optional, default=False
        Option to whether or not overwrite the csv files
    predefined_test : None or pandas.DataFrame
        Serve the case when the test set is premade (still need to match with
        the descriptors_df by index). Override test_fraction and test_points
    """
    import os
    if test_fraction:
        assert not test_points
    if training_fractions:
        assert not training_points


    if predefined_test is not None:
        target_test = predefined_test
        binned_training = bin_df(descriptors_df.drop(target_test.index),
                                 n_bins,
                                 target)

    else:
        target_test = pd.DataFrame()
        target_trainings = dict()

        if opt_in_test is not None:
            # Consider switching to search by identifiers later
            target_test = target_test.append(descriptors_df.loc[opt_in_test.index.to_list()])
            descriptors_df.drop(opt_in_test.index, inplace=True)

        binned_target = bin_df(descriptors_df,
                               n_bins,
                               target)
        binned_training = binned_target

        # First create the testing set and drop those columns from
        # the binned_training set
        if test_fraction:
            for n in range(n_bins):
                test_tmp = binned_target[f'{target}_{n}'].sample(frac=test_fraction, random_state=seed[0])
                test_tmp = test_tmp.drop(test_tmp[test_tmp['terminal_group_1']==test_tmp['terminal_group_2']].index)
                target_test = target_test.append(test_tmp)
                binned_training[f'{target}_{n}'] = binned_training[
                                               f'{target}_{n}'].drop(test_tmp.index)
        elif test_points:
            for n in range(n_bins):
                test_tmp = binned_target[f'{target}_{n}'].sample(n=int(test_points/n_bins), random_state=seed[0])
                target_test = target_test.append(test_tmp)
                binned_training[f'{target}_{n}'] = binned_training[
                                               f'{target}_{n}'].drop(test_tmp.index)
    print(f'Saving out to {output_dir}/test_set.csv')
    target_test.to_csv(f'{output_dir}/test_set.csv')

    # Then create the target_training set
    final_target = dict()
    for i in range(n_tset):
        path = f'{output_dir}/set_{i}'
        if not os.path.isdir(path):
            os.mkdir(path)
        final_target[f'set_{i}'] = dict()
        if training_fractions:
            for fraction in training_fractions:
                final_target[f'set_{i}'][f'{target}_{fraction}'] = pd.DataFrame()
                for n in range(n_bins):
                    final_target[f'set_{i}'][f'{target}_{fraction}'] = final_target[
                        f'set_{i}'][f'{target}_{fraction}'].append(binned_training[
                        f'{target}_{n}'].sample(frac=fraction, random_state=seed[i]))
                # Need overwrite option check (use os.path.isfile)
                # Raise warning (or print something out) and do nothing
                # Saving out the training csv
                filename = f'{path}/{target}_{fraction}.csv'

                if overwrite:
                    write_csv = True
                else:
                    import os
                    if not os.path.isfile(filename):
                        write_csv = True
                    else:
                        write_csv = False

                if write_csv:
                    print(f'Saving out to {filename}')
                    final_target[f'set_{i}'][f'{target}_{fraction}'].to_csv(filename)
                else:
                    continue

        elif training_points:
            for points in training_points:
                try:
                    final_target[f'set_{i}'][f'{target}_{points}'] = pd.DataFrame()
                    for n in range(n_bins):
                        final_target[f'set_{i}'][f'{target}_{points}'] = final_target[
                            f'set_{i}'][f'{target}_{points}'].append(binned_training[
                            f'{target}_{n}'].sample(n=int(points/n_bins), random_state=seed[i]))
                    # Need overwrite option check (use os.path.isfile)
                    # Raise warning (or print something out) and do nothing
                    # Saving out the training csv
                    filename = f'{path}/{target}_{points}.csv'
                except:
                    # if training_points is more than what the training set have,
                    # just create a set with all the data
                    final_target[f'set_{i}'][f'{target}_{points}'] = pd.DataFrame()
                    for n in range(n_bins):
                        final_target[f'set_{i}'][f'{target}_{points}'] = final_target[
                            f'set_{i}'][f'{target}_{points}'].append(binned_training[
                            f'{target}_{n}'].sample(frac=1, random_state=seed[i]))
                    # Need overwrite option check (use os.path.isfile)
                    # Raise warning (or print something out) and do nothing
                    # Saving out the training csv
                    filename = f'{path}/{target}_all.csv'

                if overwrite:
                    write_csv = True
                else:
                    import os
                    if not os.path.isfile(filename):
                        write_csv = True
                    else:
                        write_csv = False

                if write_csv:
                    print(f'Saving out to {filename}')
                    final_target[f'set_{i}'][f'{target}_{points}'].to_csv(filename)
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
    import os
    # Load the data
    if isinstance(data, (str, pd.DataFrame)):
        data = [data]

    # Can also consider raise an error if the input is not in the correct format

    identifiers = ['terminal_group_1',
                   'terminal_group_2',
                   'terminal_group_3',
                   'backbone',
                   'frac-1',
                   'frac-2']
    targets= ['COF', 'intercept']

    loaded_df = pd.DataFrame()
    for item in data:
        if isinstance(item, str):
            tmp_df = pd.read_csv(item, index_col=0)
        elif isinstance(item, pd.DataFrame):
            tmp_df = item
        loaded_df = loaded_df.append(tmp_df, ignore_index=True)
    '''
    to_drop = ['pc+-mean', 'pc+-diff',
               'pc+-min', 'pc+-max',
               'pc--mean', 'pc--diff',
               'pc--min', 'pc--max']
    '''
    std = ['COF-std', 'intercept-std']
    # Reduce the number of features by running them through the original
    # dimensionality_reduction by Andrew,
    # Will used the new dimensionality reduction in later version
    features = list(loaded_df.drop(targets + identifiers + std, axis=1))
    df_red_train = dimensionality_reduction(loaded_df, features,
                            filter_missing=True,
                            filter_var=True,
                            filter_corr=True,
                            missing_threshold=0.4,
                            var_threshold=0.02,
                            corr_threshold=0.9)
    red_features = list(df_red_train.drop(targets + identifiers + std, axis=1))

    X_train, y_train = df_red_train[red_features], df_red_train[target]
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
        from pathlib import Path
        # Check output parent path
        opath = Path(output_path)
        if not os.path.isdir(opath.parent):
            os.mkdir(opath.parent)
        with open(output_path+'.pickle', 'wb') as mfile:
            print('Model saved out to {}'.format(output_path+'.pickle'))
            pickle.dump(regr, mfile)
        with open(output_path+'.ptxt', 'wb') as lfile:
            print('Features saved out to {}'.format(output_path+'.ptxt'))
            pickle.dump(red_features, lfile)

    return {'model': regr,
            'features': red_features}



if __name__ == '__main__':
    overwrite = True
    for bins in [10]:
        """ Split the 50-50 mixed df and the everything df"""
        mixed5050 = pd.read_csv('../../data/raw-data/filtered_mixed-50-50.csv', index_col=0)
        everything = pd.read_csv('../../data/raw-data/filtered_everything.csv', index_col=0)
        top_sys = pd.read_csv('../../data/raw-data/opt_22_raw.csv', index_col=0)

        splitted5050_path = f'../../data/splitted-data/mixed5050/nbins-{bins}'
        splitted2575_path = f'../../data/splitted-data/mixed2575/nbins-{bins}'
        splitted_everything_path = f'../../data/splitted-data/everything/nbins-{bins}'

        for path in [splitted5050_path, splitted2575_path, splitted_everything_path]:
            if not os.path.isdir(path):
                os.mkdir(path)

        """ Only need to run this block once to create splitted data set
        Can just load data from file the next time
        """
        for target in ['COF', 'intercept']:
            split_df(descriptors_df=everything.copy(deep=True),
                                    test_fraction=0.2,
                                    opt_in_test=top_sys,
                                    training_points=[100, 200, 300, 500, 1000, 1500,
                                                     2000, 2500, 4000, 6000, 8000],
                                    n_bins=bins,
                                    target=target,
                                    output_dir=splitted_everything_path,
                                    n_tset=5,
                                    seed=[1, 10, 49, 79, 91],
                                    overwrite=overwrite)

            everything_test = pd.read_csv(f'{splitted_everything_path}/test_set.csv', index_col=0)
            predefined_5050_test = everything_test[everything_test['frac-1']==0.5]
            predefined_2575_test = everything_test[everything_test['frac-1']==0.25]
            predefined_2575_test.to_csv(f'{splitted2575_path}/test_set.csv')
            split_df(descriptors_df=mixed5050.copy(deep=True),
                                    training_points=[100, 200, 300, 500, 1000, 1500,
                                                       2000, 2500, 3000],
                                    n_bins=10,
                                    target=target,
                                    output_dir=splitted5050_path,
                                    n_tset =5,
                                    seed=[1, 10, 49, 79, 91],
                                    overwrite=overwrite,
                                    predefined_test=predefined_5050_test)

        splitted5050 = dict()
        splitted_everything = dict()

        for i in range(5):
            splitted5050[i] = {'COF': {'train_set': dict(), 'test_set': None},
                'intercept': {'train_set': dict(), 'test_set': None}}
            splitted_everything[i] = {'COF': {'train_set': dict(), 'test_set': None},
                'intercept': {'train_set': dict(), 'test_set': None}}
            e_points = [100, 200, 300, 500, 1000, 1500,
                        2000, 2500, 4000, 6000, 'all']
            m_points = [100, 200, 300, 500, 1000, 1500,
                        2000, 2500, 'all']
            for target in ['COF', 'intercept']:
                for point in m_points:
                    model_name = f'{target}_{point}'
                    filename = f'{splitted5050_path}/set_{i}/{model_name}.csv'
                    splitted5050[i][target]['train_set'][model_name] = pd.read_csv(filename, index_col=0)
                #tfilename = '{}/test_set.csv'.format(splitted5050_path, i)
                #splitted5050[i][target]['COF_test'] = pd.read_csv(tfilename, index_col=0)

            for target in ['COF', 'intercept']:
                for point in e_points:
                    model_name = f'{target}_{point}'
                    filename = f'{splitted_everything_path}/set_{i}/{model_name}.csv'
                    splitted_everything[i][target]['train_set'][model_name] = pd.read_csv(filename, index_col=0)
                #tfilename = '{}/test_set.csv'.format(splitted5050_path, i)
                #splitted_everything[i][target]['test_set'] = pd.read_csv(tfilename, index_col=0)

            """ Train the models"""
            original_csv = pd.read_csv('../../data/raw-data/original-100.csv', index_col=0)
            original_models = dict()
            mixed5050_models = dict()
            everything_models = dict()
            for target in ['COF', 'intercept']:
                # Train the original models
                omodel_path = '../models/original/{}'.format(target)
                original_models[target] = train_rf(data=original_csv,
                             target=target,
                             output_path=omodel_path,
                             overwrite=overwrite)
                # Train the mixed5050 models
                for model in splitted5050[i][target]['train_set']:
                    m5binspath = f'../models/mixed5050/nbins-{bins}'
                    if not os.path.isdir(m5binspath):
                        os.mkdir(m5binspath)
                    m5model_path = f'../models/mixed5050/nbins-{bins}/set_{i}/{model}'
                    mixed5050_models[model] = train_rf(data=splitted5050[i][target]['train_set'][model],
                                 target=target,
                                 output_path=m5model_path,
                                 overwrite=overwrite)
                # Train the everything model
                for model in splitted_everything[i][target]['train_set']:
                    ebinspath = f'../models/everything/nbins-{bins}'
                    if not os.path.isdir(ebinspath):
                        os.mkdir(ebinspath)
                    emodel_path = f'../models/everything/nbins-{bins}/set_{i}/{model}'
                    everything_models[model] = train_rf(
                                data=splitted_everything[i][target]['train_set'][model],
                                target=target,
                                output_path=emodel_path,
                                overwrite=overwrite)
