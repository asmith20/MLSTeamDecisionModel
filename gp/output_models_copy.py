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
FEATURES = ["x_real", "y_real","score"]


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


def plot_model(model, player_id, player, output_location):
    # plot the model at set points, and save
    name = player.split(' ')
    if len(name) == 2:
        filename = "{}{}.png".format(output_location, str(name[1]+str(int(player_id))))
    else:
        filename = "{}{}.png".format(output_location, str('_'.join(name[1:])+str(int(player_id))))
    x1 = np.linspace(75, 115, 25)
    x2 = np.linspace(0, 80, 25)
    #grid = np.transpose([np.tile(x1, len(x2)), np.tile(x2, len(x1))])
    xx, yy = np.meshgrid(x1, x2)
    preds = []
    try:
    	classes = model.classes_
    except AttributeError:
	#Not Enough data points so model isn't built
        return
    #for x1_val in x1:
    #    for x2_val in x2:
    #        preds.append(model.predict_proba([[x1_val, x2_val]]))
    

    #y = np.array(preds).reshape((3, 25, 25))
    k=0;
    fig1, axes = plt.subplots(nrows=3, ncols=3)
    fig1.set_size_inches(12, 10)
    figlist = []
    i=0
    k=1
    for ax in axes:

        if i == 0:
            X = np.vstack([xx.ravel(), yy.ravel(), k*np.ones(len(xx.ravel()))]).T
            y = model.predict_proba(X)
            ys = []
            ys.append(y.T[np.argwhere(model.classes_ == 'Pass'), :].reshape((25, 25)))
            ys.append(y.T[np.argwhere(model.classes_ == 'Shot'), :].reshape((25, 25)))
            ys.append(y.T[np.argwhere(model.classes_ == 'Take on'), :].reshape((25, 25)))
            k-=1

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
        ax.axis('off')
        ax.set_title(CLASSES[i], fontsize=15)

        ax.contour(xx, yy, ys[i], [.5, 1], colors='red', linewidths=2, linestyles="--")
        fig = ax.contourf(xx, yy, ys[i], vmin=0, vmax=1)
        figlist.append(fig)
        plt.ylim(0, 80)
        plt.xlim(75, 115)
        if i < 2:
            i+=1
        else:
            i=0

    m = plt.cm.ScalarMappable(cmap=cm.viridis)
    m.set_array(y)
    m.set_clim(0, 1)
    fig1.colorbar(m, ax=axes.ravel().tolist(),orientation='horizontal', ticks=np.arange(0, 1.2, .2))
    fig1.suptitle(player)
    with open(filename, 'wb') as fout:
        fig1.savefig(fout)
    return
