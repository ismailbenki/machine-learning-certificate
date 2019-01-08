import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from pandas.plotting import scatter_matrix

def convert_to_binary(X):
    return [list(map(int,x.split())) for x in X]

def convert_to_int(X):
    return [x.index(1)+1 for x in X]

def scatter(df,keys,volume=False):
    if not(volume) :
        if keys=='all':
            sm = scatter_matrix(df[:500],c=df.Cover_Type.values[:500])
        else:
            sm = scatter_matrix(df[keys][:500],c=df.Cover_Type.values[:500])

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

        for n, color in zip(range(1,8),['blue','red','green','orange','pink','yellow','purple']):

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
                name='classe ' + str(n)
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

    print('\n Matrice de confusion :\n',confusion_matrix(y_valid,predicted),'\n')
    print('\n Rapport :\n',classification_report(y_valid,predicted))

    # Evaluate the model on test set
    score = clf.score(x_valid, y_valid)

    end = time.time()

    return score,end-start
