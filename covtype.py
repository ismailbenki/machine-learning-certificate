import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from pandas.plotting import scatter_matrix
from sklearn import metrics
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

attributs_names = ['Elevation','Aspect','Slope',
                  'Horizontal_Distance_To_Hydrology',
                  'Vertical_Distance_To_Hydrology',
                  'Horizontal_Distance_To_Roadways',
                  'Hillshade_9am','Hillshade_Noon',
                  'Hillshade_3pm',
                  'Horizontal_Distance_To_Fire_Points',
                  'Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4',
                  'Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5',
                  'Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10',
                  'Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15',
                  'Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20',
                  'Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25',
                  'Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30',
                  'Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35',
                  'Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40',
                  'Cover_Type']

def convert_to_int(X):
    Y = np.array(X).astype('int')
    d = Y.shape[1]

    for i in range(d):
        Y[:,i] *= i+1

    return list(np.sum(Y,axis=1))


forest_cover_types = {1:'Spruce/Fir',2:'Lodgepole Pine',3:'Ponderosa Pine',
                      4:'Cottonwood/Willow',5:'Aspen',6:'Douglas-fir',7:'Krummholz'}

def hist_by_covtype(df,key,bins=20):

    # Make a separate list for each airline
    x1 = list(df[df['Cover_Type'] == 1][key])
    x2 = list(df[df['Cover_Type'] == 2][key])
    x3 = list(df[df['Cover_Type'] == 3][key])
    x4 = list(df[df['Cover_Type'] == 4][key])
    x5 = list(df[df['Cover_Type'] == 5][key])
    x6 = list(df[df['Cover_Type'] == 6][key])
    x7 = list(df[df['Cover_Type'] == 7][key])


    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.hist([x1, x2, x3, x4, x5,x6,x7], bins = 40, density=True,
             color = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','lightgreen','lightblue','pink'],
             label=list(forest_cover_types.values()),stacked=True)

    # Plot formatting
    plt.legend()
    plt.xlabel(key)
    plt.title('Side-by-Side Histogram for '+key)
    plt.show()



def scatter(df,keys,volume=False):
    if not(volume) :
        if keys=='all':
            sm = scatter_matrix(df[:500],c=df.Cover_Type[:500])
        else:
            sm = scatter_matrix(df[keys][:500],c=df.Cover_Type[:500])

        #Change label rotation
        [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
        [s.yaxis.label.set_rotation(45) for s in sm.reshape(-1)]

        #May need to offset label when rotating to prevent overlap of figure
        [s.get_yaxis().set_label_coords(-0.5,0.5) for s in sm.reshape(-1)]

        #Hide all ticks
        [s.set_xticks(()) for s in sm.reshape(-1)]
        [s.set_yticks(()) for s in sm.reshape(-1)]
        plt.show()
    else: #plot3d by class
        plotly.tools.set_credentials_file(username='kevinzagalo', api_key='tZ4DZDjSf7SjDoNZUZjV')
        data=[]
        cov = df.Cover_Type.values

        for n, color in zip(range(1,len(forest_cover_types)+1),['blue','red','green','orange','pink','yellow','purple']):

            xs = df[keys[0]][cov==n].values[:300]
            ys = df[keys[1]][cov==n].values[:300]
            zs = df[keys[2]][cov==n].values[:300]

            trace = go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                text=keys,
                mode='markers',
                marker=dict(
                    size=12,
                    line=dict(
                        color=color,
                        width=0.2
                    ),
                    opacity=1
                ),
                name=forest_cover_types[n]
            )
            data.append(trace)
        layout = go.Layout(margin=dict(l=0,
                                       r=0,
                                       b=0,
                                       t=0),
                            scene = dict(
                                xaxis = dict(
                                    title=keys[0]),
                                yaxis = dict(
                                    title=keys[1]),
                                zaxis = dict(
                                    title=keys[2]))
                  )

        return data,layout

def summarize(clf,x_valid,y_valid):
    start = time.time()
    predicted = clf.predict(x_valid)

    print('\n Matrice de confusion :\n',metrics.confusion_matrix(y_valid,predicted),'\n')
    print('\n Rapport :\n',metrics.classification_report(y_valid,predicted))
    print(metrics.accuracy_score(y_valid,predicted))

    # Evaluate the model on test set
    score = clf.score(x_valid, y_valid)

    end = time.time()

    return score,end-start


def redef_soil(soil,dict_x,bi_class=False):
    soil_x = []
    keys = list(dict_x.keys())

    for x in np.array(soil):
        soil_x0 = [0]*len(keys)
        count = 0
        for index,key in zip(range(len(keys)),keys):
            if bi_class:
                if (sum(x[dict_x[key]]) == 1) and (count<2):
                    soil_x0[index] = 1
                    count += 1
                elif (count == 2):
                    break
            else:
                if sum(x[dict_x[key]]) == 1:
                    soil_x0[index] = 1
                    break
        soil_x.append(soil_x0)

    return soil_x

def redef_aspect(Aspect):
    Aspect[Aspect == 360] = 0
    nesw = []
    x = np.arange(9)
    bornes = np.ones(9) * 180/8 + x * 180/4
    bornes = np.insert(bornes,0,0)
    bornes[-1] = 360

    for aspect in Aspect:
        nesw_ = [0]*8
        for i in range(9):
            if (bornes[i] <= aspect and aspect < bornes[i+1]):
                nesw_[i%8] = 1
                break
        nesw.append(nesw_)

    return nesw

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(model,title, X, y,n_jobs = 1, ylim = None, cv = None,train_sizes = np.linspace(0.1, 1, 5)):

    # Figrue parameters
    plt.figure(figsize=(10,8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')

    train_sizes, train_score, test_score = learning_curve(model, X, y, cv = cv, n_jobs=n_jobs, train_sizes=train_sizes)

    # Calculate mean and std
    train_score_mean = np.mean(train_score, axis=1)
    train_score_std = np.std(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    test_score_std = np.std(test_score, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_score_mean - train_score_std, train_score_mean + train_score_std,\
                    alpha = 0.1, color = 'r')
    plt.fill_between(train_sizes, test_score_mean - test_score_std, test_score_mean + test_score_std,\
                    alpha = 0.1, color = 'g')

    plt.plot(train_sizes, train_score_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_score_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc = "best")
    plt.show()


dict_family = {'Cathedral':[0],'Vanet':[1,4,5],'Haploborolis':[2],'Ratake':[1,3],'Gothic':[6],'Supervisor':[7],
               'Troutville':[8],'Bullwark':[9,10,12],'Legault':[11,28,29],'Catamount':[9,10,12,25,30,31],'Pachic':[13],
               'unspecified':[14],'Cryaquolis':[15,16,18],'Rogert':[17],'Cryaquepts':[19,34],'Cryaquolls':[19,20,22],
               'Leighcan':[20,21,22,23,24,26,27,30,31,32,37,38],'Como':[28,29],'Cryorthents':[33,36,38,39],
               'Cryumbrepts':[34,35,36],'Bross':[35],'Moran':[37,38,39]}

dict_group = {'stony':[0,1,5,6,8,11,17,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],
               'rubbly':[2,3,4,9,10,12],'other':[7,13,14,15,16,18,19,20,21,22]}

def dist(x,y):
    return np.sqrt(np.array(x)**2 + np.array(y)**2)


def covtype_load_data(url_str,pca=False,pca_label=4):
    print('Loading file...')
    # df_covtype : DataFrame de toutes les données non traitées triées par attribut
    df_covtype = pd.read_csv(url_str,
                             header=None,
                             names=attributs_names,
                             compression='gzip')

    # On met toute les données numérique au type int
    for name in attributs_names:
        df_covtype[name] = df_covtype[name].astype('int')

    # labels : array des étiquettes des types de forêts
    covtypes = list(set(df_covtype.Cover_Type))
    print('Multi-class :',forest_cover_types)

    print('Features :')

    print('   Wilderness_Area...')

    wilderness = pd.concat([df_covtype[name] for name in attributs_names[10:14]],axis=1).values
    df_covtype = df_covtype.drop(attributs_names[10:14],axis=1)
    df_covtype['Wilderness_Area'] = convert_to_int(wilderness)

    print('   Deleting Soil_Type...')

    soil = pd.concat([df_covtype[name] for name in attributs_names[14:54]],axis=1).values
    df_covtype = df_covtype.drop(attributs_names[14:54],axis=1)
    df_covtype['Soil_Type'] = convert_to_int(soil)

    soil_family = redef_soil(soil,dict_family,bi_class=True)
    df_covtype['Soil_Family'] = convert_to_int(soil_family)
    print('   Soil_Family')

    soil_group = redef_soil(soil,dict_group)
    df_covtype['Soil_Group'] = convert_to_int(soil_group)
    print('   Soil_Group')

    aspect_group = redef_aspect(df_covtype.Aspect.values)
    df_covtype['Aspect_Group'] = convert_to_int(aspect_group)
    print('   Aspect_Group : (N,NE,E,SE,S,SW,W,NW)')

    df_covtype['sqrt_Fire'] = np.sqrt(df_covtype.Horizontal_Distance_To_Fire_Points.values)
    print('   sqr_Fire = sqrt(Horizontal_Distance_To_Fire_Points)')

    df_covtype['sqrt_Roadways'] = np.sqrt(df_covtype.Horizontal_Distance_To_Roadways.values)
    print('   sqr_Roadways = sqrt(Horizontal_Distance_To_Roadways)')

    df_covtype['sin_Aspect'] = np.sin(df_covtype.Aspect.values * np.pi/180)
    print('   sin_Aspect = sin(Aspect)')

    df_covtype['cos_Slope'] = np.cos(df_covtype.Slope.values * np.pi/180)
    print('   cos_Slope = cos(Slope)')

    df_covtype['Distance_To_Hydrology'] = dist(df_covtype.Vertical_Distance_To_Hydrology.values,
                                           df_covtype.Horizontal_Distance_To_Hydrology.values)
    print('   Distance_To_Hydrology = sqrt(Vertical_Distance_To_Hydrology^2 + Horizontal_Distance_To_Hydrology^2)')

    df_covtype['Hillshade_mean'] = df_covtype[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']].mean(axis=1)
    print('   Hillshade_mean = 1/3 * (Hillshade_9am+Hillshade_Noon+Hillshade_3pm)')
    print('   Hillshade_3pm')
    print('   Elevation')

    target = df_covtype.Cover_Type.values

    data0 = df_covtype.drop(['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
                         'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',
                         'Slope','Hillshade_9am','Hillshade_Noon','Aspect','Aspect_Group',
                         'Cover_Type','Wilderness_Area','Soil_Type','Soil_Family','Soil_Group'],axis=1).values

    if pca:
        print('projecting pca...')
        pca = PCA().fit(data0[target == pca_label,:])
        data0 = np.dot(data0,pca.components_) # projection des données sur les axes prinipaux de la pca

    N,d = data0.shape
    N,w = np.array(wilderness).shape
    N,sf = np.array(soil_family).shape
    N,sg = np.array(soil_group).shape
    N,a = np.array(aspect_group).shape
    data = np.zeros((N,d+w+sf+sg+a))

    # Normalisation
    print('Normalization...')
    for i in range(d):
        x = data0[:,i]
        data0[:,i] = (x-np.mean(x))/sum(x)

    # On remet les données qualitative de sorte à pouvoir les exploiter
    binary_data = np.concatenate([wilderness,soil_family,soil_group,aspect_group],axis=1)
    data = np.concatenate([data0,binary_data],axis=1)

    print('Done.')

    return data,target
