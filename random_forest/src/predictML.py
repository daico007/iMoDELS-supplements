"""Import neccessary packages"""
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn import ensemble, linear_model, metrics, model_selection
import pickle
import os

from atools_ml.descriptors import rdkit_descriptors


def calculate_raw_descriptors(h_smiles, ch3_smiles, ind_descriptors):
    '''Retrieve h_smiles and ch3_smiles descriptors from ind_descriptor.

    If parameters do not exist in ind_descriptors, calculate the parameters
    from scratch (averagin over 1000 trials)

    Parameters
    ----------
    h_smiles : str
        h_smiles string of the interested chemistry
    ch3_smiles : str
        ch3_smiles string of the interested chemistry
    ind_descriptors : pd.Dataframe
        Dataframe which contains the reference descriptors

    Returns
    -------
    tg_descriptors : dict
    '''
    import pandas as pd

    if isinstance(ind_descriptors, str):
        ind_descriptors = pd.read_csv(ind_descriptors, index_col=0)
    assert isinstance(ind_descriptors, pd.DataFrame)

    final_desc_h_tg = dict()
    final_desc_ch3_tg = dict()

    tg_descriptors = dict()

    if (h_smiles not in ind_descriptors) or ch3_smiles not in ind_descriptors:
        print(f'Calculating chemical descriptors for {h_smiles} and {ch3_smiles}')
        for i in range(1000):
            tmp_desc_h_tg = rdkit_descriptors(h_smiles)
            tmp_desc_ch3_tg = rdkit_descriptors(ch3_smiles,
                include_h_bond=True, ch3_smiles=ch3_smiles)

            for key in tmp_desc_h_tg:
                if key in final_desc_h_tg:
                    final_desc_h_tg[key].append(tmp_desc_h_tg[key])
                else:
                    final_desc_h_tg[key] = [tmp_desc_h_tg[key]]
            for key in tmp_desc_ch3_tg:
                if key in final_desc_ch3_tg:
                    final_desc_ch3_tg[key].append(tmp_desc_ch3_tg[key])
                else:
                    final_desc_ch3_tg[key] = [tmp_desc_ch3_tg[key]]
        for key in final_desc_h_tg:
            final_desc_h_tg[key] = np.mean(final_desc_h_tg[key])
        for key in final_desc_ch3_tg:
            final_desc_ch3_tg[key] = np.mean(final_desc_ch3_tg[key])

        tg_descriptors[h_smiles] = final_desc_h_tg
        tg_descriptors[ch3_smiles] = final_desc_ch3_tg
        result = pd.DataFrame.from_dict(tg_descriptors)
        ind_descriptors.insert(loc=-1, column=h_smiles, value=result[h_smiles])
        ind_descriptors.insert(loc=-1, column=ch3_smiles, values=result[ch3_smiles])
        ind_descriptors.to_csv('updated_ind_descriptors.csv')
    else:
        print(f'{h_smiles} and {ch3_smiles} already exist in ind_descriptors')
        result = ind_descriptors[[h_smiles, ch3_smiles]]

    return result


def consolidate_descriptors(top_smiles, top_frac,
                            bot_smiles, bot_frac,
                            desc_df):
    '''Consolidate h_smiles and ch3_smiles descriptors

    Parameters
    ----------
    top_smiles : list
        List of smiles strings of the top monolayer, each chemistry
        need to be represented by 2 elements in a tuple, i.e. (h_smiles, ch3_smiles)
    top_frac : list
        List of fraction corresponding to the list of smiles in the top
        monolyaer
    bot_smiles : list
        List of smiles strings of the bottom monolayer, each chemistry
        need to be represented by 2 elements in a tuple, i.e. (h_smiles, ch3_smiles)
    bot_frac : list
        List of fraction corresponding to the list of smiles in the bottom
        monolayer
    desc_df : pd.DataFrame
        DataFrame which contains individual descriptors of SMILES string

    Returns
    -------
    df_predict : pd.Series
        Consolidated list of parameters representing the system
    '''

    assert len(top_smiles) == len(top_frac)
    assert len(bot_smiles) == len(bot_frac)
    assert sum(top_frac) == 1
    assert sum(bot_frac) == 1

    if isinstance(desc_df, str):
        desc_df = pd.read_csv(desc_df, index_col=0)
    assert isinstance(desc_df, pd.DataFrame)

    #to_drop = ['pc+-mean', 'pc+-min', 'pc--mean', 'pc--min']
    with open('data/raw-data/feature-clusters.json', 'r') as f:
        clusters = json.load(f) # this is a dict
    shape_features = clusters['shape'] # a list from the clusters dict
    top_desc = {'h': dict(), 'ch3': dict()}
    bot_desc = {'h': dict(), 'ch3': dict()}
    for key in desc_df.index:
        top_desc['h'][key] = top_desc['ch3'][key] = 0
        bot_desc['h'][key] = bot_desc['ch3'][key] = 0
        for i in range(len(top_smiles)):
            top_desc['h'][key] += desc_df[top_smiles[i][0]][key] * top_frac[i]
            top_desc['ch3'][key] += desc_df[top_smiles[i][1]][key] * top_frac[i]
        for j in range(len(bot_smiles)):
            bot_desc['h'][key] += desc_df[bot_smiles[j][0]][key] * bot_frac[j]
            bot_desc['ch3'][key] += desc_df[bot_smiles[j][1]][key] * bot_frac[j]

    desc_h_df = pd.DataFrame([top_desc['h'], bot_desc['h']])
    desc_ch3_df = pd.DataFrame([top_desc['ch3'], bot_desc['ch3']])

    desc_df = []
    for i, df in enumerate([desc_h_df, desc_ch3_df]):
        if i == 1:
            hbond_tb = max(df['hdonors'][0], df['hacceptors'][1]) \
                       if all((df['hdonors'][0], df['hacceptors'][1])) \
                       else 0
            hbond_bt = max(df['hdonors'][1], df['hacceptors'][0]) \
                       if all((df['hdonors'][1], df['hacceptors'][0])) \
                       else 0
            hbonds = hbond_tb + hbond_bt
            df.drop(['hdonors', 'hacceptors'], 'columns', inplace=True)
        else:
            hbonds = 0
        means = df.mean()
        mins = df.min()
        means = means.rename({label: '{}-mean'.format(label)
                              for label in means.index})
        mins = mins.rename({label: '{}-min'.format(label)
                            for label in mins.index})
        desc_tmp = pd.concat([means, mins])
        desc_tmp['hbonds'] = hbonds
        #desc_tmp.drop(labels=to_drop, inplace=True)
        desc_df.append(desc_tmp)

    df_h_predict = desc_df[0]
    df_ch3_predict = desc_df[1]
    df_h_predict = pd.concat([
        df_h_predict.filter(like=feature) for feature in shape_features], axis=0)
    df_ch3_predict.drop(labels=df_h_predict.keys(), inplace=True)

    df_h_predict_mean = df_h_predict.filter(like='-mean')
    df_h_predict_min = df_h_predict.filter(like='-min')
    df_ch3_predict_mean = df_ch3_predict.filter(like='-mean')
    df_ch3_predict_min = df_ch3_predict.filter(like='-min')

    df_predict = pd.concat([df_h_predict_mean, df_h_predict_min,
                            df_ch3_predict_mean, df_ch3_predict_min,
                            df_ch3_predict[['hbonds']]])

    return df_predict


def predict_properties(top_smiles, top_frac,
                       bot_smiles, bot_frac,
                       COF_model, COF_features,
                       F0_model, F0_features,
                       ind_desc):
    '''Consolidate h_smiles and ch3_smiles descriptors

    Parameters
    ----------
    top_smiles : list
        List of smiles strings of the top monolayer, each chemistry
        need to be represented by 2 elements in a tuple, i.e. (h_smiles, ch3_smiles)
    top_frac : list
        List of fraction corresponding to the list of smiles in the top
        monolyaer
    bot_smiles : list
        List of smiles strings of the bottom monolayer, each chemistry
        need to be represented by 2 elements in a tuple, i.e. (h_smiles, ch3_smiles)
    bot_frac : list
        List of fraction corresponding to the list of smiles in the bottom
        monolayer
    COF_model : RandomForestRegressor
        Random forest model to predict COF property
    COF_features : list
        List of features used for the COF model
    F0_model : RandomForestRegressor
        Random forest model to predict F0 property
    F0_features : list
        List of features used for the F0 model
    ind_desc : pd.DataFrame
        DataFrame which contains individual descriptors of SMILES string

    Returns
    -------
    {'COF': predicted_COF,
     'F0': predicted_F0}
    '''
    smiles_set = set(top_smiles + bot_smiles)
    desc_df = pd.DataFrame()
    for smiles in smiles_set:
        processed_df = calculate_raw_descriptors(h_smiles=smiles[0],
                                                 ch3_smiles=smiles[1],
                                                 ind_descriptors=ind_desc)
        if not any(desc_df):
            desc_df = processed_df
        else:
            desc_df.insert(loc=-1, column=smiles[0], value=processed_df)
            desc_df.insert(loc=-1, column=smiles[1], value=processed_df)

    sys_df = consolidate_descriptors(top_smiles=top_smiles,
                                     top_frac=top_frac,
                                     bot_smiles=bot_smiles,
                                     bot_frac=bot_frac,
                                     desc_df=desc_df)

    if isinstance(COF_model, str):
        with open(COF_model, 'rb') as m:
            COF_model = pickle.load(m)
    if isinstance(COF_features, str):
        with open(COF_features, 'rb') as f:
            COF_features = pickle.load(f)

    if isinstance(F0_model, str):
        with open(F0_model, 'rb') as m:
            F0_model = pickle.load(m)
    if isinstance(F0_features, str):
        with open(F0_features, 'rb') as f:
            F0_features = pickle.load(f)

    filtered_COF_df = sys_df.filter(COF_features)
    filtered_F0_df = sys_df.filter(F0_features)

    predicted_COF = COF_model.predict(np.asarray(filtered_COF_df).reshape(1, -1))
    predicted_F0 = F0_model.predict(np.asarray(filtered_F0_df).reshape(1, -1))

    print(f'Predicted COF: {predicted_COF[0]}')
    print(f'Predicted F0: {predicted_F0[0]}')

    return {'COF': predicted_COF[0],
            'F0': predicted_F0[0]}


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
        test_df = pd.read_csv(test_df, index_col=0)
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
