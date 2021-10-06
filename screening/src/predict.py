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
        ind_path = ind_descriptors
        ind_descriptors = pd.read_csv(ind_descriptors, index_col=0)
    else:
        ind_path = None
    assert isinstance(ind_descriptors, pd.DataFrame)

    final_desc_h_tg = dict()
    final_desc_ch3_tg = dict()

    tg_descriptors = dict()

    if (h_smiles not in ind_descriptors) or (ch3_smiles not in ind_descriptors):
        print(f'Calculating chemical descriptors for {h_smiles} and {ch3_smiles}')
        for i in range(1000):
            # Change to calculate hbonds regardless
            tmp_desc_h_tg = rdkit_descriptors(h_smiles, include_h_bond=True, ch3_smiles=h_smiles)
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
        if h_smiles not in ind_descriptors.columns:
            ind_descriptors.insert(loc=len(ind_descriptors.columns), column=h_smiles, value=result[h_smiles])
        if ch3_smiles not in ind_descriptors.columns:
            ind_descriptors.insert(loc=len(ind_descriptors.columns), column=ch3_smiles, value=result[ch3_smiles])
        if ind_path:
            ind_descriptors.to_csv(ind_path)
    else:
        print(f'{h_smiles} and {ch3_smiles} already exist in ind_descriptors')
        result = ind_descriptors[[h_smiles, ch3_smiles]]

    return result


def consolidate_descriptors(top_smiles, top_frac,
                            bot_smiles, bot_frac,
                            desc_df,
                            feature_clusters):
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
    with open(feature_clusters, 'r') as f:
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
                       ind_desc,
                       feature_clusters):
    '''Consolidate SMILES strings descriptors and calculate COF and F0 values

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

    top_smiles = [tuple(pair) for pair in top_smiles]
    bot_smiles = [tuple(pair) for pair in bot_smiles]
    smiles_set = set(top_smiles + bot_smiles)
    desc_df = pd.DataFrame()
    for smiles in smiles_set:
        processed_df = calculate_raw_descriptors(h_smiles=smiles[0],
                                                 ch3_smiles=smiles[1],
                                                 ind_descriptors=ind_desc)
        if not any(desc_df):
            desc_df = processed_df
        else:
            for smile in smiles:
                if smile not in desc_df.columns:
                    desc_df.insert(loc=len(desc_df.columns), column=smile, value=processed_df[smile])

    sys_df = consolidate_descriptors(top_smiles=top_smiles,
                                     top_frac=top_frac,
                                     bot_smiles=bot_smiles,
                                     bot_frac=bot_frac,
                                     desc_df=desc_df,
                                     feature_clusters=feature_clusters)

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

