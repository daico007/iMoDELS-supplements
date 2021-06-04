import pickle
import json
import matplotlib
from matplotlib import pyplot as plt
import sklearn
import pandas as pd
import os
from matplotlib.legend_handler import HandlerBase

#https://stackoverflow.com/questions/47391702/matplotlib-making-a-colored-markers-legend-from-scratch
class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup,xdescent, ydescent,
                        width, height, fontsize,trans):
        return [plt.Line2D([width/2], [height/2.],ls="",
                       marker=tup[1],color=tup[0], transform=trans)]

#https://stackoverflow.com/questions/52303660/iterating-markers-in-plots/52303895#52303895
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def plot_feature_importances(model,
                    train_df,
                    target,
                    features,
                    feature_clusters,
                    output,
                    top_n=7):
    """Plot a horizontal bar graph of the features importances of each model

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor or str
        The model or path to the model
    train_df : str or df
        The training df or path to the training df used for this model
    target : str ('COF' or 'intercept')
        Target of the predictive model
    features : list or str
        The list of features that comes with the model.
        Ordered is important.
    feature_clusters : dict or str
        Feature clusters used to group and color
        individual color bar
    output : str
        Path of the output plot
    top_n : int, optional, default=7
        Save one version with only top n sfeatures

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
    if isinstance(train_df, str):
        train_df = pd.read_csv(train_df)

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
    cats_marker = {'complexity': 'd',
                   'qdist': 'o',
                   'shape': '^',
                   'size': 's'}

    feature_importances_dict = {
        'name': features,
        'value': model.feature_importances_.tolist(),
        'color': [cats_color[mod_clusters[feature]]
                for feature in features],
        'marker': [cats_marker[mod_clusters[feature]]
                for feature in features],
        'corr': [train_df.corr().at[target, feature]
                for feature in features]}

    feature_importances_df = pd.DataFrame(feature_importances_dict)

    # Unranked
    plt.figure(figsize=(12, 14))

    plt.hlines(feature_importances_df['name'],
               xmin=0,
               xmax=feature_importances_df['value'],
               color=['red' if val>=0 else 'skyblue'
               for val in feature_importances_df['corr']],
               zorder=1)

    mscatter(feature_importances_df['value'],
             feature_importances_df['name'],
             m=feature_importances_df['marker'],
             color=feature_importances_df['color'],
             edgecolor='black',
             zorder=2)

    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.savefig(f'{output}_unranked.pdf',
             dpi=500, bbox_inches='tight')

    # Ranked version
    feature_importances_df.sort_values(
                                by='value',
                                inplace=True)

    plt.figure(figsize=(12, 14))

    plt.hlines(feature_importances_df['name'],
               xmin=0,
               xmax=feature_importances_df['value'],
               color=['red' if val>=0 else 'skyblue'
               for val in feature_importances_df['corr']],
               zorder=1)

    mscatter(feature_importances_df['value'],
             feature_importances_df['name'],
             m=feature_importances_df['marker'],
             color=feature_importances_df['color'],
             edgecolor='black',
             zorder=2)

    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.savefig(f'{output}_ranked.pdf',
             dpi=500, bbox_inches='tight')

    # Only top n

    plt.figure(figsize=(10, 8))

    plt.hlines(feature_importances_df['name'].tail(top_n),
               xmin=0,
               xmax=feature_importances_df['value'].tail(top_n),
               color=['red' if val>=0 else 'skyblue'
               for val in feature_importances_df['corr'].tail(top_n)],
               zorder=1)

    mscatter(feature_importances_df['value'].tail(top_n),
             feature_importances_df['name'].tail(top_n),
             m=feature_importances_df['marker'].tail(top_n),
             color=feature_importances_df['color'].tail(top_n),
             edgecolor='black',
             zorder=2)

    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.savefig(f'{output}_top{top_n}.pdf',
             dpi=500, bbox_inches='tight')

    plt.close('all')
    return None

def plot_double_feature_importances(
                COF_model, COF_features, COF_train_df,
                intercept_model, intercept_features, intercept_train_df,
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
        Path of the output plot

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
    if isinstance(COF_train_df, str):
        COF_train_df = pd.read_csv(COF_train_df, index_col=0)
    if isinstance(intercept_model, str):
        with open(intercept_model, 'rb') as f:
            intercept_model = pickle.load(f)
    if isinstance(intercept_features, str):
        with open(intercept_features, 'rb') as f:
            intercept_features = pickle.load(f)
    if isinstance(intercept_train_df, str):
        intercept_train_df = pd.read_csv(intercept_train_df, index_col=0)
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
    cats_marker = {'complexity': 'd',
                   'qdist': 'o',
                   'shape': '^',
                   'size': 's'}

    COF_feature_importances_dict = {
        'name': COF_features,
        'value': COF_model.feature_importances_.tolist(),
        'color': [cats_color[mod_clusters[feature]]
                for feature in COF_features],
        'marker': [cats_marker[mod_clusters[feature]]
                for feature in COF_features],
        'corr': [COF_train_df.corr().at['COF', feature]
                for feature in COF_features]}

    COF_feature_importances_df = pd.DataFrame(COF_feature_importances_dict)
    intercept_feature_importances_dict = {
        'name': intercept_features,
        'value': intercept_model.feature_importances_.tolist(),
        'color': [cats_color[mod_clusters[feature]]
                for feature in intercept_features],
        'marker': [cats_marker[mod_clusters[feature]]
                for feature in intercept_features],
        'corr': [intercept_train_df.corr().at['intercept', feature]
                for feature in intercept_features]}
    intercept_feature_importances_df = pd.DataFrame(intercept_feature_importances_dict)

    # Unranked

    plt.figure(figsize=(24, 14))

    plt.subplot(1, 2, 1)
    plt.hlines(COF_feature_importances_df['name'],
               xmin=0,
               xmax=COF_feature_importances_df['value'],
               color=['red' if val>=0 else 'skyblue'
               for val in COF_feature_importances_df['corr']],
               zorder=1)

    mscatter(COF_feature_importances_df['value'],
             COF_feature_importances_df['name'],
             m=COF_feature_importances_df['marker'],
             color=COF_feature_importances_df['color'],
             edgecolor='black',
             zorder=2)

    plt.title('COF Model', fontsize=28, weight='bold')
    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.subplot(1, 2, 2)
    plt.hlines(intercept_feature_importances_df['name'],
               xmin=0,
               xmax=intercept_feature_importances_df['value'],
               color=['red' if val>=0 else 'skyblue'
               for val in intercept_feature_importances_df['corr']],
               zorder=1)

    mscatter(intercept_feature_importances_df['value'],
             intercept_feature_importances_df['name'],
             m=intercept_feature_importances_df['marker'],
             color=intercept_feature_importances_df['color'],
             edgecolor='black',
             zorder=2)

    plt.title('F$_0$ Model', fontsize=28, weight='bold')
    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.savefig(f'{output}_unranked.pdf',
             dpi=500, bbox_inches='tight')

    # Ranked
    COF_feature_importances_df.sort_values(
                                    by='value',
                                    inplace=True)
    intercept_feature_importances_df.sort_values(
                                    by='value',
                                    inplace=True)

    plt.figure(figsize=(24, 14))

    plt.subplot(1, 2, 1)
    plt.hlines(COF_feature_importances_df['name'],
               xmin=0,
               xmax=COF_feature_importances_df['value'],
               color=['red' if val>=0 else 'skyblue'
               for val in COF_feature_importances_df['corr']],
               zorder=1)

    mscatter(COF_feature_importances_df['value'],
             COF_feature_importances_df['name'],
             m=COF_feature_importances_df['marker'],
             color=COF_feature_importances_df['color'],
             edgecolor='black',
             zorder=2)

    plt.title('COF Model', fontsize=28, weight='bold')
    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.subplot(1, 2, 2)
    plt.hlines(intercept_feature_importances_df['name'],
               xmin=0,
               xmax=intercept_feature_importances_df['value'],
               color=['red' if val>=0 else 'skyblue'
               for val in intercept_feature_importances_df['corr']],
               zorder=1)

    mscatter(intercept_feature_importances_df['value'],
             intercept_feature_importances_df['name'],
             m=intercept_feature_importances_df['marker'],
             color=intercept_feature_importances_df['color'],
             edgecolor='black',
             zorder=2)

    plt.title('F$_0$ Model', fontsize=28, weight='bold')
    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.savefig(f'{output}_ranked.pdf',
             dpi=500, bbox_inches='tight')

    # Top_n

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.hlines(COF_feature_importances_df['name'].tail(top_n),
               xmin=0,
               xmax=COF_feature_importances_df['value'].tail(top_n),
               color=['red' if val>=0 else 'skyblue'
               for val in COF_feature_importances_df['corr'].tail(top_n)],
               zorder=1)

    mscatter(COF_feature_importances_df['value'].tail(top_n),
             COF_feature_importances_df['name'].tail(top_n),
             m=COF_feature_importances_df['marker'].tail(top_n),
             color=COF_feature_importances_df['color'].tail(top_n),
             edgecolor='black',
             zorder=2)

    plt.title('COF Model', fontsize=28, weight='bold')
    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.subplot(1, 2, 2)
    plt.hlines(intercept_feature_importances_df['name'].tail(top_n),
               xmin=0,
               xmax=intercept_feature_importances_df['value'].tail(top_n),
               color=['red' if val>=0 else 'skyblue'
               for val in intercept_feature_importances_df['corr'].tail(top_n)],
               zorder=1)

    mscatter(intercept_feature_importances_df['value'].tail(top_n),
             intercept_feature_importances_df['name'].tail(top_n),
             m=intercept_feature_importances_df['marker'].tail(top_n),
             color=intercept_feature_importances_df['color'].tail(top_n),
             edgecolor='black',
             zorder=2)

    plt.title('F$_0$ Model', fontsize=28, weight='bold')
    plt.xlabel('Relative Importances')
    plt.ylabel('Features')
    plt.xlim(0)
    leg = plt.legend([('black', 'd'), ('yellow', 'o'),
                      ('cornflowerblue', '^'), ('red', 's')],
                    ['Complexity','Charge distribution','Shape','Size'],
                    handler_map={tuple:MarkerHandler()},
                    loc=4)

    for legobj in leg.legendHandles:
        legobj.set_markeredgecolor('black')
        legobj.set_markeredgewidth(1)

    plt.savefig(f'{output}_top{top_n}.pdf',
             dpi=500, bbox_inches='tight')


    plt.close('all')
    return None

def plot_simulated_predicted(predicted_data,
                             output,
                             bound_lines=None,
                             alpha=0.1):
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


    alpha = alpha
    plt.figure(figsize=(10, 8))
    plt.scatter(results['x'], results['y'], alpha=alpha, marker='o')
    if target == 'COF':
        plt.xlabel('{} (Predicted)'.format(target))
        plt.ylabel('{} (Simulated)'.format(target))
        plt.xlim(0.085, 0.2)
        plt.ylim(0.085, 0.2)
        plt.text(x=0.16, y=0.1, s=f'r$^2$={results["r_square"]:.3f}', fontsize=28)
    elif target == 'intercept':
        plt.xlim(-1, 9)
        plt.ylim(-1, 9)
        plt.xlabel('F$_0$ (Predicted)')
        plt.ylabel('F$_0$ (Simulated)')
        plt.text(x=5.5, y=0.3, s=f'r$^2$={results["r_square"]:.3f}', fontsize=28)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, alpha=0.2)

    # Draw bound line if need be
    if bound_lines:
        yuppers = [x*(1+bound_lines) for x in xpoints]
        ylowers = [x*(1-bound_lines) for x in xpoints]
        plt.plot(xpoints, yuppers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=0.2)
        plt.plot(xpoints, ylowers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=0.2)

    plt.savefig(f'{output}.pdf', dpi=500, bbox_inches='tight')

    plt.close('all')
    return None

def plot_double_simulated_predicted(
                COF_data, intercept_data,
                output,
                bound_lines=None,
                alpha=0.1):
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


    plt.figure(figsize=(20, 8))
    alpha = alpha

    # Plot COF
    plt.subplot(1, 2, 1)
    plt.title('COF Model', fontsize=28, weight='bold')
    plt.xlabel('COF (Predicted)')
    plt.ylabel('COF (Simulated)')
    plt.scatter(results['COF']['x'], results['COF']['y'], alpha=alpha, marker='o')
    plt.text(x=0.16, y=0.1, s=f'r$^2$={results["COF"]["r_square"]:.3f}', fontsize=28)
    plt.xlim(0.085, 0.2)
    plt.ylim(0.085, 0.2)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, alpha=0.2)

    if bound_lines:
        yuppers = [x*(1+bound_lines) for x in xpoints]
        ylowers = [x*(1-bound_lines) for x in xpoints]
        plt.plot(xpoints, yuppers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=0.2)
        plt.plot(xpoints, ylowers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=0.2)

    # Plot intercept
    plt.subplot(1, 2, 2)
    plt.title('F$_{0}$ Model', fontsize=28, weight='bold')
    plt.xlabel('F$_{0}$ (Predicted), nN')
    plt.ylabel('F$_{0}$ (Simulated), nN')
    plt.scatter(results['intercept']['x'], results['intercept']['y'], alpha=alpha, marker='o')
    plt.text(x=5.5, y=0.3, s=f'r$^2$={results["intercept"]["r_square"]:.3f}', fontsize=28)
    plt.xlim(-1, 9)
    plt.ylim(-1, 9)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, alpha=0.2)

    if bound_lines:
        yuppers = [x*(1+bound_lines) for x in xpoints]
        ylowers = [x*(1-bound_lines) for x in xpoints]
        plt.plot(xpoints, yuppers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=0.2)
        plt.plot(xpoints, ylowers, linestyle='--',
            color='k', lw=3, scalex=False,
            scaley=False, alpha=0.2)

    plt.savefig(f'{output}.pdf', dpi=500, bbox_inches='tight')
    
    plt.close('all')
    return None

if __name__ == '__main__':
    fc_path = 'feature-clusters.json'
    alpha=0.1
    for bins in [10]:
        for i in range(5):
            oresults_path = f'../predicted-results/original/nbins-{bins}'
            mresults_path = f'../predicted-results/mixed5050/nbins-{bins}/set_{i}'
            eresults_path = f'../predicted-results/everything/nbins-{bins}/set_{i}'

            for path in [oresults_path, mresults_path, eresults_path]:
                if not os.path.isdir(f'{path}/plots'):
                    os.mkdir(f'{path}/plots')

            omodels_path = f'../models/original'
            mmodels_path = f'../models/mixed5050/nbins-{bins}/set_{i}'
            emodels_path = f'../models/everything/nbins-{bins}/set_{i}'

            otrains_path = f'../../data/raw-data'
            mtrains_path = f'../../data/splitted-data/mixed5050/nbins-{bins}/set_{i}'
            etrains_path = f'../../data/splitted-data/everything/nbins-{bins}/set_{i}'

            epoints = [100, 200, 300, 500, 1000, 1500,
                        2000, 2500, 4000, 6000, 'all']
            mpoints = [100, 200, 300, 500, 1000, 1500,
                        2000, 2500, 'all']
            test_sets = ['5050', '2575', 'everything']

            """Plot feature importances"""
            top_n = 8

            # First handle the original models
            for target in ['COF', 'intercept']:
                m_file = f'{omodels_path}/{target}.pickle'
                f_file = f'{omodels_path}/{target}.ptxt'
                t_file = f'{otrains_path}/original-100.csv'
                output = f'{oresults_path}/plots/fi_{target}'

                plot_feature_importances(model=m_file,
                                         features=f_file,
                                         train_df=t_file,
                                         target=target,
                                         feature_clusters=fc_path,
                                         output=output,
                                         top_n=top_n)
            plot_double_feature_importances(
                                COF_model=f'{omodels_path}/COF.pickle',
                                COF_features=f'{omodels_path}/COF.ptxt',
                                COF_train_df=f'{otrains_path}/original-100.csv',
                                intercept_model=f'{omodels_path}/intercept.pickle',
                                intercept_features=f'{omodels_path}/intercept.ptxt',
                                intercept_train_df=f'{otrains_path}/original-100.csv',
                                feature_clusters=fc_path,
                                output=f'{oresults_path}/plots/dfi_original',
                                top_n=top_n)

            # Then handle the 5050 model
            for points in mpoints:
                for target in ['COF', 'intercept']:
                    m_file = f'{mmodels_path}/{target}_{points}.pickle'
                    f_file = f'{mmodels_path}/{target}_{points}.ptxt'
                    t_file = f'{mtrains_path}/{target}_{points}.csv'
                    output = f'{mresults_path}/plots/fi_{target}_{points}'
                    plot_feature_importances(model=m_file,
                                             train_df=t_file,
                                             target=target,
                                             features=f_file,
                                             feature_clusters=fc_path,
                                             output=output,
                                             top_n=top_n)
                plot_double_feature_importances(
                            COF_model=f'{mmodels_path}/COF_{points}.pickle',
                            COF_features=f'{mmodels_path}/COF_{points}.ptxt',
                            COF_train_df=f'{mtrains_path}/COF_{points}.csv',
                            intercept_model=f'{mmodels_path}/intercept_{points}.pickle',
                            intercept_features=f'{mmodels_path}/intercept_{points}.ptxt',
                            intercept_train_df=f'{mtrains_path}/intercept_{points}.csv',
                            feature_clusters=fc_path,
                            output=f'{mresults_path}/plots/dfi_mixed5050_{points}',
                            top_n=top_n)

            # Last handle the everything model
            for points in epoints:
                for target in ['COF', 'intercept']:
                    m_file = f'{emodels_path}/{target}_{points}.pickle'
                    f_file = f'{emodels_path}/{target}_{points}.ptxt'
                    t_file = f'{etrains_path}/{target}_{points}.csv'
                    output = f'{eresults_path}/plots/fi_{target}_{points}'
                    plot_feature_importances(model=m_file,
                                             features=f_file,
                                             train_df=t_file,
                                             target=target,
                                             feature_clusters=fc_path,
                                             output=output,
                                             top_n=top_n)
                plot_double_feature_importances(
                            COF_model=f'{emodels_path}/COF_{points}.pickle',
                            COF_features=f'{emodels_path}/COF_{points}.ptxt',
                            COF_train_df=f'{etrains_path}/COF_{points}.csv',
                            intercept_model=f'{emodels_path}/intercept_{points}.pickle',
                            intercept_features=f'{emodels_path}/intercept_{points}.ptxt',
                            intercept_train_df=f'{etrains_path}/intercept_{points}.csv',
                            feature_clusters=fc_path,
                            output=f'{eresults_path}/plots/dfi_everything_{points}',
                            top_n=top_n)

            """Plot simulated vs predicted"""
            bound_lines = 0.15
            # First handle the original models
            for tset in test_sets:
                for target in ['COF', 'intercept']:
                    oresult = f'{oresults_path}/{target}_on_{tset}.json'
                    output = f'{oresults_path}/plots/ps_{target}_on_{tset}'
                    plot_simulated_predicted(
                                predicted_data=oresult,
                                output=output,
                                bound_lines=bound_lines,
                                alpha=alpha)

                plot_double_simulated_predicted(
                             COF_data=f'{oresults_path}/COF_on_{tset}.json',
                             intercept_data=f'{oresults_path}/intercept_on_{tset}.json',
                             output=f'{oresults_path}/plots/dps_original_on_{tset}',
                             bound_lines=bound_lines,
                             alpha=alpha)

            # Then handle the 5050 model
            for points in mpoints:
                for tset in test_sets:
                    for target in ['COF', 'intercept']:
                        mresult = f'{mresults_path}/{target}_{points}_on_{tset}.json'
                        output = f'{mresults_path}/plots/ps_{target}_{points}_on_{tset}'
                        plot_simulated_predicted(
                                    predicted_data=mresult,
                                    output=output,
                                    bound_lines=bound_lines,
                                    alpha=alpha)

                    plot_double_simulated_predicted(
                                 COF_data=f'{mresults_path}/COF_{points}_on_{tset}.json',
                                 intercept_data=f'{mresults_path}/intercept_{points}_on_{tset}.json',
                                 output=f'{mresults_path}/plots/dps_mixed5050_{points}_on_{tset}',
                                 bound_lines=bound_lines,
                                 alpha=alpha)

            # Last handle the everything set
            for points in epoints:
                for tset in test_sets:
                    for target in ['COF', 'intercept']:
                        eresult = f'{eresults_path}/{target}_{points}_on_{tset}.json'
                        output = f'{eresults_path}/plots/ps_{target}_{points}_on_{tset}'

                        plot_simulated_predicted(
                                    predicted_data=eresult,
                                    output=output,
                                    bound_lines=bound_lines,
                                    alpha=alpha)

                    plot_double_simulated_predicted(
                                 COF_data=f'{eresults_path}/COF_{points}_on_{tset}.json',
                                 intercept_data=f'{eresults_path}/intercept_{points}_on_{tset}.json',
                                 output=f'{eresults_path}/plots/dps_everything_{points}_on_{tset}',
                                 bound_lines=bound_lines,
                                 alpha=alpha)
