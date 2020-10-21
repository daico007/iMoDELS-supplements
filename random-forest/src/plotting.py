import pickle
import json
import matplotlib
from matplotlib import pyplot as plt
import sklearn
import pandas as pd
import os

def plot_feature_importances(model,
                    features,
                    feature_clusters,
                    output,
                    top_n=7):
    """Plot a horizontal bar graph of the features importances of each model

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor or str
        The model or path to the model
    features : list or str
        The list of features that comes with the model.
        Ordered is important.
    feature_clusters : dict or str
        Feature clusters used to group and color
        individual color bar
    output : str
        Path of the output plot
    top_n : int, optional, default=7
        Save one version with only top n features

    Returns
    -------
    Plots saved to disk
    """
    if isinstance(model, str):
        with open(model, 'rb') as f:
            model = pickle.load(f)
    if isinstance(features, str):
        with open(features, 'rb') as f:
            features = pickle.load(f)
    if isinstance(feature_clusters, str):
        with open(feature_clusters, 'r') as f:
            feature_clusters = json.load(f)
    mod_clusters = dict()
    for cluster in feature_clusters:
        for feature in feature_clusters[cluster]:
            if feature == 'hbonds':
                mod_clusters[feature] = cluster
            else:
                mod_clusters[feature+'-mean']= cluster
                mod_clusters[feature+'-min']= cluster

    cats = feature_clusters.keys()
    cats_color = {'complexity': 'black',
                 'qdist': 'yellow',
                 'shape': 'blue',
                 'size': 'red'}

    feature_importances_dict = {
        'name': features,
        'value': model.feature_importances_.tolist(),
        'color': [cats_color[mod_clusters[feature]]
                for feature in features]}
    feature_importances_df = pd.DataFrame(feature_importances_dict)

    # Unranked
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 10))

    plt.barh(feature_importances_df['name'],
             feature_importances_df['value'],
             color=feature_importances_df['color'],
             height=0.5)
    if target == 'COF':
        plt.title('COF Feature Importances')
    elif target == 'intercept':
        plt.title('F$_0$ Feature Importance')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
               for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution', 'Shape', 'Size'], prop={'size': 15})
    plt.savefig('{}_unranked.png'.format(output),
             dpi=500, bbox_inches='tight')

    # Ranked version
    feature_importances_df.sort_values(
                                by='value',
                                inplace=True)
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 10))
    plt.barh(feature_importances_df['name'],
             feature_importances_df['value'],
             color=feature_importances_df['color'],
             height=0.5)
    if target == 'COF':
        plt.title('COF Feature Importances')
    elif target == 'intercept':
        plt.title('F$_0$ Feature Importance')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
               for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution', 'Shape', 'Size'], prop={'size': 15})
    plt.savefig('{}_ranked.png'.format(output),
             dpi=500, bbox_inches='tight')

    # Only top n
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 10))
    plt.barh(feature_importances_df['name'].tail(top_n),
             feature_importances_df['value'].tail(top_n),
             color=feature_importances_df['color'].tail(top_n),
             height=0.5)
    if target == 'COF':
        plt.title('COF Feature Importances')
    elif target == 'intercept':
        plt.title('F$_0$ Feature Importance')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
               for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution', 'Shape', 'Size'], prop={'size': 15})
    plt.savefig('{}_top{}.png'.format(output, top_n),
             dpi=500, bbox_inches='tight')

    plt.close('all')
    return None

def plot_double_feature_importances(
                COF_model, COF_features,
                intercept_model, intercept_features,
                feature_clusters, output, top_n=7):
    """Plot both COF and intercept models feature importances on the same plot

    Parameters
    ----------
    COF_model : sklearn.ensemble.RandomForestRegressor or str
        The COF model or path to the model.
    COF_features : list or str
        The list of features that comes with the COF model.
        Ordered is important.
    intercept_model : sklearn.ensemble.RandomForestRegressor or str
        The intercept model or path to the model.
    intercept_features : list or str
        The list of features that comes with the intercept mode.
        Ordered is important.
    feature_clusters : dict or str
        Feature clusters used to group and color
        individual color bar
    top_n : int, optional, default=7
        Save one version with only top n features
    output : str
        Path of the ouput plot

    Returns
    -------
    Plots saved to disk
    """
    if isinstance(COF_model, str):
        with open(COF_model, 'rb') as f:
            COF_model = pickle.load(f)
    if isinstance(COF_features, str):
        with open(COF_features, 'rb') as f:
            COF_features = pickle.load(f)
    if isinstance(intercept_model, str):
        with open(intercept_model, 'rb') as f:
            intercept_model = pickle.load(f)
    if isinstance(intercept_features, str):
        with open(intercept_features, 'rb') as f:
            intercept_features = pickle.load(f)
    if isinstance(feature_clusters, str):
        with open(feature_clusters, 'r') as f:
            feature_clusters = json.load(f)

    mod_clusters = dict()
    for cluster in feature_clusters:
        for feature in feature_clusters[cluster]:
            if feature == 'hbonds':
                mod_clusters[feature] = cluster
            else:
                mod_clusters[feature+'-mean'] = cluster
                mod_clusters[feature+'-min'] = cluster
    cats = feature_clusters.keys()
    cats_color = {'complexity': 'black',
                  'qdist': 'yellow',
                  'shape': 'blue',
                  'size': 'red'}

    COF_feature_importances_dict = {
        'name': COF_features,
        'value': COF_model.feature_importances_.tolist(),
        'color': [cats_color[mod_clusters[feature]]
                for feature in COF_features]}
    COF_feature_importances_df = pd.DataFrame(COF_feature_importances_dict)
    intercept_feature_importances_dict = {
        'name': intercept_features,
        'value': intercept_model.feature_importances_.tolist(),
        'color': [cats_color[mod_clusters[feature]]
                for feature in intercept_features]}
    intercept_feature_importances_df = pd.DataFrame(intercept_feature_importances_dict)

    # Unranked
    plt.style.use('ggplot')
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.barh(COF_feature_importances_df['name'],
             COF_feature_importances_df['value'],
             color=COF_feature_importances_df['color'],
             height=0.5)
    plt.title('COF feature importances')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
             for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution',
                          'Shape', 'Size'], prop={'size': 15})

    plt.subplot(1, 2, 2)
    plt.barh(intercept_feature_importances_df['name'],
             intercept_feature_importances_df['value'],
             color=intercept_feature_importances_df['color'],
             height=0.5)
    plt.title('F$_0$ feature importances')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
             for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution',
                          'Shape', 'Size'], prop={'size': 15})
    plt.savefig('{}_unranked.png'.format(output),
             dpi=500, bbox_inches='tight')

    # Ranked
    COF_feature_importances_df.sort_values(
                                    by='value',
                                    inplace=True)
    intercept_feature_importances_df.sort_values(
                                    by='value',
                                    inplace=True)
    plt.style.use('ggplot')
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.barh(COF_feature_importances_df['name'],
             COF_feature_importances_df['value'],
             color=COF_feature_importances_df['color'],
             height=0.5)
    plt.title('COF feature importances')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
             for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution',
                          'Shape', 'Size'], prop={'size': 15})

    plt.subplot(1, 2, 2)
    plt.barh(intercept_feature_importances_df['name'],
             intercept_feature_importances_df['value'],
             color=intercept_feature_importances_df['color'],
             height=0.5)
    plt.title('F$_0$ feature importances')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
             for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution',
                          'Shape', 'Size'], prop={'size': 15})
    plt.savefig('{}_ranked.png'.format(output),
             dpi=500, bbox_inches='tight')

    # Top_n
    plt.style.use('ggplot')
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.barh(COF_feature_importances_df['name'].tail(top_n),
             COF_feature_importances_df['value'].tail(top_n),
             color=COF_feature_importances_df['color'].tail(top_n),
             height=0.5)
    plt.title('COF feature importances')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
             for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution',
                          'Shape', 'Size'], prop={'size': 15})

    plt.subplot(1, 2, 2)
    plt.barh(intercept_feature_importances_df['name'].tail(top_n),
             intercept_feature_importances_df['value'].tail(top_n),
             color=intercept_feature_importances_df['color'].tail(top_n),
             height=0.5)
    plt.title('F$_0$ feature importances')
    handles = [plt.Rectangle((0,0),1,1, color=cats_color[cat])
             for cat in cats]
    plt.legend(handles, ['Complexity', 'Charge Distribution',
                          'Shape', 'Size'], prop={'size': 15})
    plt.savefig('{}_top{}.png'.format(output, top_n),
             dpi=500, bbox_inches='tight')

    plt.close('all')
    return None

def plot_simulated_predicted(predicted_data,
                             output,
                             bound_lines=None,
                             ):
    """Plot the simulated vs predicted plot of the generated dta file

    Paratermeters
    -------------
    predicted_data : dict or str
        Path to the data file (in json format)
    output : str
        Path of the output plot
    bound_lines : float or None, optional, default=None

    Returns
    -------
    plot save to file

    """
    if isinstance(predicted_data, str):
        with open(predicted_data, 'r') as f:
            predicted_data = json.load(f)

    # Parse plot title
    # Can play with this more later
    output_file = os.path.split(output)[1]

    target = list(predicted_data.keys())[0]
    results = {'x': list(),
               'y': list(),
               'r_square': predicted_data[target].pop('r_square')}
    for idx in predicted_data[target]:
        results['x'].append(
            predicted_data[target][str(idx)]['predicted-{}'.format(target)])
        results['y'].append(
            predicted_data[target][str(idx)]['simulated-{}'.format(target)])

    plt.style.use('default')
    alpha = 0.2

    plt.scatter(results['x'], results['y'], alpha=alpha)
    if target == 'COF':
        plt.xlabel('{} (Predicted)'.format(target), fontsize=15, weight='bold')
        plt.ylabel('{} (Simulated)'.format(target), fontsize=15, weight='bold')
        plt.title('COF - Predicted vs Simulated', fontsize=20)
        plt.xlim(0.085, 0.2)
        plt.ylim(0.085, 0.2)
    elif target == 'intercept':
        plt.title('F$_0$ - Predicted vs Simulated', fontsize=20)
        plt.xlim(-1, 9)
        plt.ylim(-1, 9)
        plt.xlabel('F$_0$ (Predicted)', fontsize=15, weight='bold')
        plt.ylabel('F$_0$ (Simulated)', fontsize=15, weight='bold')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, alpha=alpha)

    # Draw bound line if need be
    if bound_lines:
        yuppers = [x*(1+bound_lines) for x in xpoints]
        ylowers = [x*(1-bound_lines) for x in xpoints]
        plt.plot(xpoints, yuppers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=alpha)
        plt.plot(xpoints, ylowers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=alpha)

    plt.savefig('{}.png'.format(output), dpi=500, bbox_inches='tight')

    plt.close('all')
    return None

def plot_double_simulated_predicted(
                COF_data, intercept_data, output, bound_lines=None):
    """Plot both COF and intercept models simulated vs predicted on the same plot
    """
    if isinstance(COF_data, str):
        with open(COF_data, 'r') as f:
            COF_data = json.load(f)
    if isinstance(intercept_data, str):
        with open(intercept_data, 'r') as f:
            intercept_data = json.load(f)

    # Parse plot title
    # Can play with this more later
    output_file = os.path.split(output)[1]

    data = {**COF_data,
            **intercept_data}
    results = {'COF' : {
                'x' : list(),
                'y' : list(),
                'r_square' : COF_data['COF'].pop('r_square')},
               'intercept' : {
                'x' : list(),
                'y' : list(),
                'r_square' : intercept_data['intercept'].pop('r_square')}}

    for target in data:
        for idx in data[target]:
            results[target]['x'].append(
                data[target][str(idx)]['predicted-{}'.format(target)])
            results[target]['y'].append(
                data[target][str(idx)]['simulated-{}'.format(target)])

    plt.style.use('default')
    plt.figure(figsize=(10, 8))
    alpha = 0.2

    # Plot COF
    plt.subplot(221)
    plt.title('Predicted vs Simulated - COF', fontsize=20)
    plt.xlabel('COF (Predicted)', fontsize=15, weight='bold')
    plt.ylabel('COF (Simulated)', fontsize=15, weight='bold')
    plt.scatter(results['COF']['x'], results['COF']['y'], alpha=alpha)
    plt.xlim(0.085, 0.2)
    plt.ylim(0.085, 0.2)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, alpha=alpha)

    if bound_lines:
        yuppers = [x*(1+bound_lines) for x in xpoints]
        ylowers = [x*(1-bound_lines) for x in xpoints]
        plt.plot(xpoints, yuppers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=alpha)
        plt.plot(xpoints, ylowers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=alpha)

    # Plot intercept
    plt.subplot(222)
    plt.title('F$_0$ - Predicted vs Simulated', fontsize=20)
    plt.xlabel('F$_{0}$ (Predicted), nN', fontsize=15, weight='bold')
    plt.ylabel('F$_{0}$ (Simulated), nN', fontsize=15, weight='bold')
    plt.scatter(results['intercept']['x'], results['intercept']['y'], alpha=alpha)
    plt.xlim(-1, 9)
    plt.ylim(-1, 9)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, alpha=alpha)

    if bound_lines:
        yuppers = [x*(1+bound_lines) for x in xpoints]
        ylowers = [x*(1-bound_lines) for x in xpoints]
        plt.plot(xpoints, yuppers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=alpha)
        plt.plot(xpoints, ylowers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=alpha)

    plt.savefig('{}.png'.format(output), dpi=500, bbox_inches='tight')
    
    plt.close('all')
    return None

if __name__ == '__main__':
    fc_path = 'feature-clusters.json'

    oresults_path = '../predicted-results/original'
    mresults_path = '../predicted-results/mixed5050'
    eresults_path = '../predicted-results/everything'

    omodels_path = '../models/original'
    mmodels_path = '../models/mixed5050'
    emodels_path = '../models/everything'

    mfractions = [0.05, 0.1, 0.2,
                  0.3, 0.5, 0.7, 1]
    efractions = [0.01, 0.02, 0.03,
                 0.05, 0.1, 0.2, 0.3,
                 0.5, 0.7, 1]

    test_sets = ['5050', '2575', 'everything']

    """Plot feature importances"""
    top_n = 7

    # First handle the original models
    for target in ['COF', 'intercept']:
        m_file = '{}/{}.pickle'.format(
                                omodels_path,
                                target)
        f_file = '{}/{}.ptxt'.format(
                                omodels_path,
                                target)
        output = '{}/plots/fi_{}'.format(
                                oresults_path,
                                target)

        plot_feature_importances(model=m_file,
                                 features=f_file,
                                 feature_clusters=fc_path,
                                 output=output,
                                 top_n=top_n)
    plot_double_feature_importances(
                        COF_model='{}/COF.pickle'.format(omodels_path),
                        COF_features='{}/COF.ptxt'.format(omodels_path),
                        intercept_model='{}/intercept.pickle'.format(omodels_path),
                        intercept_features='{}/intercept.ptxt'.format(omodels_path),
                        feature_clusters=fc_path,
                        output='{}/plots/dfi_original'.format(oresults_path),
                        top_n=top_n)

    # Then handle the 5050 model
    for fraction in mfractions:
        for target in ['COF', 'intercept']:
            m_file = '{}/{}_{}.pickle'.format(mmodels_path, target, fraction)
            f_file = '{}/{}_{}.ptxt'.format(mmodels_path, target, fraction)
            output = '{}/plots/fi_{}_{}'.format(mresults_path, target, fraction)
            plot_feature_importances(model=m_file,
                                     features=f_file,
                                     feature_clusters=fc_path,
                                     output=output,
                                     top_n=top_n)
        plot_double_feature_importances(
                    COF_model='{}/COF_{}.pickle'.format(mmodels_path, fraction),
                    COF_features='{}/COF_{}.ptxt'.format(mmodels_path, fraction),
                    intercept_model='{}/intercept_{}.pickle'.format(mmodels_path, fraction),
                    intercept_features='{}/intercept_{}.ptxt'.format(mmodels_path, fraction),
                    feature_clusters=fc_path,
                    output='{}/plots/dfi_mixed5050_{}'.format(mresults_path,
                                                                  fraction),
                    top_n=top_n)

    # Last handle the 2575 model
    for fraction in efractions:
        for target in ['COF', 'intercept']:
            m_file = '{}/{}_{}.pickle'.format(emodels_path, target, fraction)
            f_file = '{}/{}_{}.ptxt'.format(emodels_path, target, fraction)
            output = '{}/plots/fi_{}_{}'.format(eresults_path, target, fraction)
            plot_feature_importances(model=m_file,
                                     features=f_file,
                                     feature_clusters=fc_path,
                                     output=output,
                                     top_n=top_n)
        plot_double_feature_importances(
                    COF_model='{}/COF_{}.pickle'.format(emodels_path, fraction),
                    COF_features='{}/COF_{}.ptxt'.format(emodels_path, fraction),
                    intercept_model='{}/intercept_{}.pickle'.format(emodels_path, fraction),
                    intercept_features='{}/intercept_{}.ptxt'.format(emodels_path, fraction),
                    feature_clusters=fc_path,
                    output='{}/plots/dfi_everything_{}'.format(eresults_path,
                                                                   fraction),
                    top_n=top_n)

    """Plot simulated vs predicted"""
    bound_lines = 0.15
    # First handle the original models
    for tset in test_sets:
        for target in ['COF', 'intercept']:
            oresult = '{}/{}_on_{}.json'.format(
                        oresults_path,
                        target,
                        tset)
            output = '{}/plots/ps_{}_on_{}'.format(
                        oresults_path,
                        target,
                        tset)
            plot_simulated_predicted(
                        predicted_data=oresult,
                        output=output,
                        bound_lines=bound_lines)

        plot_double_simulated_predicted(
                    COF_data='{}/COF_on_{}.json'.format(oresults_path, tset),
                     intercept_data='{}/intercept_on_{}.json'.format(
                                                                oresults_path,
                                                                tset),
                     output='{}/plots/dps_original_on_{}'.format(
                                                                oresults_path,
                                                                tset),
                     bound_lines=bound_lines)

    # Then handle the 5050 model
    for fraction in mfractions:
        for tset in test_sets:
            for target in ['COF', 'intercept']:
                mresult = '{}/{}_{}_on_{}.json'.format(
                            mresults_path,
                            target,
                            fraction,
                            tset)
                output = '{}/plots/ps_{}_{}_on_{}'.format(
                            mresults_path,
                            target,
                            fraction,
                            tset)
                plot_simulated_predicted(
                            predicted_data=mresult,
                            output=output,
                            bound_lines=bound_lines)

            plot_double_simulated_predicted(
                          COF_data='{}/COF_{}_on_{}.json'.format(
                                                        mresults_path,
                                                        fraction,
                                                        tset),
                         intercept_data='{}/intercept_{}_on_{}.json'.format(
                                                        mresults_path,
                                                        fraction,
                                                        tset),
                         output='{}/plots/dps_mixed5050_{}_on_{}'.format(
                                                        mresults_path,
                                                        fraction,
                                                        tset),
                         bound_lines=bound_lines)

    # Last handle the everything set
    for fraction in efractions:
        for tset in test_sets:
            for target in ['COF', 'intercept']:
                eresult = '{}/{}_{}_on_{}.json'.format(
                            eresults_path,
                            target,
                            fraction,
                            tset)
                output = '{}/plots/ps_{}_{}_on_{}'.format(
                            eresults_path,
                            target,
                            fraction,
                            tset)
                plot_simulated_predicted(
                            predicted_data=eresult,
                            output=output,
                            bound_lines=bound_lines)

            plot_double_simulated_predicted(
                          COF_data='{}/COF_{}_on_{}.json'.format(
                                                        eresults_path,
                                                        fraction,
                                                        tset),
                         intercept_data='{}/intercept_{}_on_{}.json'.format(
                                                        eresults_path,
                                                        fraction,
                                                        tset),
                         output='{}/plots/dps_everything_{}_on_{}'.format(
                                                        eresults_path,
                                                        fraction,
                                                        tset),
                         bound_lines=bound_lines)
