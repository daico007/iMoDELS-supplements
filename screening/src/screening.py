import pandas as pd
import sklearn
import pickle
import itertools
import json
import sys

from .predict import predict_properties
import time

def h_to_ch3_smiles(h_smiles):
    """
    Parameters
    ----------
    h_smiles : str
        h_smiles that need to be converted

    Returns
    -------
    ch3_smiles :
        converted ch3_smiles
    """
    carbons = ["c", "C"]

    ch3_smiles = None
    if len(h_smiles) == 1:
        ch3_smiles = "C" + h_smiles
    elif h_smiles[0] in carbons:
        ch3_smiles = "C" + h_smiles
    elif h_smiles[-1] in carbons:
        ch3_smiles = h_smiles + "c"
    elif h_smiles[1] in carbons:
        ch3_smiles = f"C({h_smiles[0]}){h_smiles[1:]}"
    elif h_smiles[-2] in carbons:
        ch3_smiles = f"{h_smiles[:-1]}({h_smiles[-1]})C"
    else:
        if "C" in h_smiles or "c" in h_smiles:
            for idx, letter in enumerate(h_smiles):
                if letter in carbons:
                    if (h_smiles[idx-1] != "#" and
                        h_smiles[idx+1] != "#" and
                        not (h_smiles[idx+1]=="=" and h_smiles[idx-1]=="=")):
                        ch3_smiles = f"{h_smiles[:idx+1]}(C){h_smiles[idx+1:]}"
                        break

    return ch3_smiles

def filter_metal(df):
    """Filter out h_smiles with metal elements

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be modified

    Returns
    -------
    df : pd.DataFrame
        The filtered DataFrame
    """
    metals = ['Li','Be','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
         'Zn','Ga','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In',
         'Sn','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er',
         'Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
         'Po','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm',
         'Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv',
          'li','be','mg','al','k','ca','sc','ti','v','cr','mn','fe','co','ni','cu',
         'zn','ga','rb','sr','y','zr','nb','mo','tc','ru','rh','pd','ag','cd','in',
         'sn','cs','ba','la','ce','pr','nd','pm','sm','eu','gd','tb','dy','ho','er',
         'tm','yb','lu','hf','ta','w','re','os','ir','pt','au','hg','tl','pb','bi',
         'po','fr','ra','ac','th','pa','u','np','pu','am','cm','bk','cf','es','fm',
         'md','no','lr','rf','db','sg','bh','hs','mt','ds','rg','cn','nh','fl','mc','lv']
    df = df[~df.h_smiles.str.contains('|'.join(metals), na=False)]
    return df

def add_ch3_smiles(df):
    """Add a ch3_smiles column to the dataframe, assume the h_smiles column exists

    Parameters
    ----------
    df : pd.DataFrame
        The data to be added on.
    Returns
    -------
    df : pd.DataFrame
        Modified DataFrame
    """
    if not "ch3_smiles" in df:
        df["ch3_smiles"] = ""
    for idx, row in df.iterrows():
        if isinstance(row["h_smiles"], str):
            df["ch3_smiles"][idx] = h_to_ch3_smiles(row["h_smiles"])
    return df

def filter_empty_ch3_smiles(df):
    """ Filter out row in dataframe that does not have ch3_smiles entry

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be modified

    Returns
    -------
    filtered_df : pd.DataFrame
        Filtered DataFrame
    """
    filtered_df = pd.DataFrame()
    not_processed = pd.DataFrame()
    for idx, row in df.iterrows():
        if not row["ch3_smiles"]:
            not_processed = not_processed.append(row)
        else:
            filtered_df = filtered_df.append(row)
    return filtered_df, not_processed

def init_design_space(df, top_components=1, top_fracs=[1], bot_components=1, bot_fracs=[1]):
    """Create dictionary with system design information

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all chemistries of interest
    top_components : int, optional, default=1
        Number of components in the top monolayer
    top_fracs : list of float, optional, default=[1]
        Fraction of components in the top monolayer.
        Lengths of list must match with top_components and sum up to 1.
    bot_components : int, optional, default=1
        Number of compoentns in the bottom monolayer.
    bot_fracs : list of float, optional, default=[1]
        Fractions of components in the bottom monolayer.
        Lenghts of list must match with bot_components and sum up to 1.

    Returns
    -------
    designs : dict
        Dictionary of the design space. Key of each entry has the format of
        "system_i" where i indicate the order of which design is created.
        (Note: i follow 0 indexing)
    """

    combinations = itertools.product(list(df.index), repeat=top_components + bot_components)

    sys_count = 0
    sys_dict = dict()
    for combination in combinations:
        sys_name = f'system_{sys_count}'
        sys_count += 1

        top_smiles=list()
        for i in range(top_components):
            top_smiles.append((df['h_smiles'][combination[i]], df['ch3_smiles'][combination[i]]))

        bot_smiles=list()
        for j in range(top_components, top_components+bot_components):
            bot_smiles.append((df['h_smiles'][combination[j]], df['ch3_smiles'][combination[j]]))

        sys_dict[sys_name] = {'top_smiles': top_smiles,
                              'top_fracs': top_fracs,
                              'bot_smiles': bot_smiles,
                              'bot_fracs': bot_fracs}

    return sys_dict

def screening_tribological_properties(designs_dict, COF_model, COF_features, F0_model, F0_features, ind_desc, feature_clusters):
    """Load in ML model and perform screening tribologiacl properties

    Parameters
    ----------
    designs_dict : dict
        Dictionary containing all the systems and their design specs
    COF_model : str or sklearn model
        Path to the pickled ML model or the model itself
    COF_features : str or list
        Path to the features json or list of features itself
    F0_model : str or sklearn model
        Path to the pickled ML model or the model itself
    F0_features : str or list
        Path to the features json or list of feature itself
    ind_desc : str
        Path to a csv file with descriptors of individual chemistry
    feature_clusters : str
        Path to the feature clusters json
    Returns
    -------
    designs_dict : dict
        The same dictionary with added predicted properties
    """
    from copy import deepcopy
    results=deepcopy(designs_dict)
    for sys_name, sys_props in designs_dict.items():
        try:
            predicted = predict_properties(top_smiles=sys_props["top_smiles"],
                                       top_frac=sys_props["top_fracs"],
                                       bot_smiles=sys_props["bot_smiles"],
                                       bot_frac=sys_props["bot_fracs"],
                                       COF_model=COF_model,
                                       COF_features=COF_features,
                                       F0_model=F0_model,
                                       F0_features=F0_features,
                                       ind_desc=ind_desc,
                                       feature_clusters=feature_clusters)
            results[sys_name].update(predicted)
            results[sys_name]["status"] = "pass"
        except:
            results[sys_name]["status"] = f"{sys.exc_info()[0]}"

    return results

if __name__ == "__main__":
    ''' Move the filtering step to a notebook
    # Load in the target DataFrame
    df = pd.read_csv('./csv/ChemBL_4to99.csv', sep=";")
    df.rename({"Smiles": "h_smiles"}, axis=1, inplace=True)

    # Filter h_smiles with metallic components
    df = filter_metal(df)

    # Create ch3_smiles based on the provided h_smiles
    df = add_ch3_smiles(df)

    # Filter failed ch3_smiles conversion
    df, failed = filter_empty_ch3_smiles(df)

    # Create design space, store in dictionary
    designs_dict = init_design_space(df)
    '''

    # Create design space from a filtered csv, store in dictionary
    df = pd.read_csv('./csv/filtered.csv', index_col=0)
    designs_dict = init_design_space(df)
    # Open COF and F0 models to be used
    COF_paths = {'model': './../random_forest/models/everything/nbins-10/set_0/COF_all.pickle',
                 'features': './../random_forest/models/everything/nbins-10/set_0/COF_all.ptxt'}
    F0_paths = {'model': './../random_forest/models/everything/nbins-10/set_0/intercept_all.pickle',
                'features': './../random_forest/models/everything/nbins-10/set_0/intercept_all.ptxt'}

    with open(COF_paths['model'], 'rb') as cm:
        COF_model = pickle.load(cm)
    with open(COF_paths['features'], 'rb') as cf:
        COF_features = pickle.load(cf)

    with open(F0_paths['model'], 'rb') as fm:
        F0_model = pickle.load(fm)
    with open(F0_paths['features'], 'rb') as ff:
        F0_features = pickle.load(ff)


    # Screen for tribological using ML models
    start = time.time()
    results = screening_tribological_properties(designs_dict,
                                                COF_model=COF_model,
                                                COF_features=COF_features,
                                                F0_model=F0_model,
                                                F0_features=F0_features,
                                                ind_desc='./csv/descriptors-ind.csv',
                                                feature_clusters='./../data/raw-data/feature-clusters.json')

    end = time.time()
    results['elapsed-time'] = end - start
    with open('./screening_results/ChemBL_4to99_1top1bot.json', 'w') as f:
        json.dump(results, f, indent=4)
