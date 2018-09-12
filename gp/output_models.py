from __future__ import division
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel,RationalQuadratic
import matplotlib
matplotlib.use('Agg')
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib.patches import Arc


# shouldnt duplicate these either but its a hackathon so
N_CLASSES = 3
CLASSES = ["Pass", "Shot", "Take on"]
FEATURES = ["x_real", "y_real"]


def build_model(player_data):
    # build a player specific model
    X = player_data[FEATURES]
    y = player_data.event_type
    gp_opt = GaussianProcessClassifier(kernel=ConstantKernel() + RationalQuadratic(length_scale=1,alpha=1.5) + WhiteKernel(noise_level=1), n_restarts_optimizer=2)
    try:
        if y.nunique() == 3:
            gp_opt.fit(X, y)
    except ValueError as e:
        return
    
    return gp_opt


def plot_model(model1,model2,model3, player_id, player, output_location):
    # plot the model at set points, and save
    name = player.split(' ')
    filename = "{}{}.png".format(output_location, str('_'.join(name[0:])))
    x1 = np.linspace(75, 115, 25)
    x2 = np.linspace(0, 80, 25)
    #grid = np.transpose([np.tile(x1, len(x2)), np.tile(x2, len(x1))])
    xx, yy = np.meshgrid(x1, x2)
    preds = []
    try:
        classes1 = model1.classes_
        classes2 = model2.classes_
        classes3 = model3.classes_
    except AttributeError:
	#Not Enough data points so model isn't built
        return
    fig1, axes = plt.subplots(nrows=3, ncols=3)
    fig1.set_size_inches(10, 16)
    figlist = []
    k=1
    for i in range(3):
        X = np.vstack([xx.ravel(), yy.ravel()]).T
        ys = []
        if i==0:
            y = model1.predict_proba(X)
            ys.append(y.T[np.argwhere(model1.classes_ == 'Pass'), :].reshape((25, 25)))
            ys.append(y.T[np.argwhere(model1.classes_ == 'Shot'), :].reshape((25, 25)))
            ys.append(y.T[np.argwhere(model1.classes_ == 'Take on'), :].reshape((25, 25)))

        elif i==1:
            y = model2.predict_proba(X)
            ys.append(y.T[np.argwhere(model2.classes_ == 'Pass'), :].reshape((25, 25)))
            ys.append(y.T[np.argwhere(model2.classes_ == 'Shot'), :].reshape((25, 25)))
            ys.append(y.T[np.argwhere(model2.classes_ == 'Take on'), :].reshape((25, 25)))
        else:
            y = model3.predict_proba(X)
            ys.append(y.T[np.argwhere(model3.classes_ == 'Pass'), :].reshape((25, 25)))
            ys.append(y.T[np.argwhere(model3.classes_ == 'Shot'), :].reshape((25, 25)))
            ys.append(y.T[np.argwhere(model3.classes_ == 'Take on'), :].reshape((25, 25)))

        k -= 1

        for j in range(3):
            ax = axes[i,j]

            #Pitch Outline & Centre Line
            ax.plot([75,75],[0,80], color="black")
            ax.plot([75,115],[80,80], color="black")
            ax.plot([115,115],[80,0], color="black")
            ax.plot([115,75],[0,0], color="black")

            #Right Penalty Area
            ax.plot([115,97],[62,62],color="black")
            ax.plot([97,97],[62,18],color="black")
            ax.plot([97,115],[18,18],color="black")

            #Right 6-yard Box
            ax.plot([115,109],[50,50],color="black")
            ax.plot([109,109],[50,30],color="black")
            ax.plot([109,115],[30,30],color="black")

            #Prepare Circles
            rightPenSpot = plt.Circle((103,40),0.8,color="black")

            #Draw Circles
            ax.add_patch(rightPenSpot)

            ax.set_aspect('equal')

            #Prepare Arcs
            rightArc = Arc((103,40),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color="black")

            #Draw Arcs
            ax.add_patch(rightArc)
            ax.set_title(CLASSES[j], fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            if j==0:
                if i==0:
                    ax.set_ylabel('Ahead',fontsize=14)
                elif i==1:
                    ax.set_ylabel('Tied',fontsize=14)
                else:
                    ax.set_ylabel('Behind',fontsize=14)

            ax.contour(xx, yy, ys[j], [.5, 1], colors='red', linewidths=2, linestyles="--")
            fig = ax.contourf(xx, yy, ys[j], vmin=0, vmax=1)
            figlist.append(fig)
            plt.ylim(0, 80)
            plt.xlim(75, 115)

    m = plt.cm.ScalarMappable(cmap=cm.viridis)
    m.set_array(y)
    m.set_clim(0, 1)
    fig1.colorbar(m, ax=axes.ravel().tolist(),orientation='horizontal', ticks=np.arange(0, 1.2, .2))
    fig1.suptitle(player+str(' 2018'),fontsize=24)
    with open(filename, 'wb') as fout:
        fig1.savefig(fout)
    return
