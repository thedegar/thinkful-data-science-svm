# coding=utf-8
#####################################################
# Tyler Hedegard
# 7/27/16
# Thinkful Data Science
# SVM
#####################################################

from sklearn import datasets
from sklearn import svm
from matplotlib.colors import ListedColormap
import numpy as np


iris = datasets.load_iris()
setosa = iris.data[0:50]
versicolor = iris.data[50:100]
verginica = iris.data[100:150]
flowers = [setosa, versicolor, verginica]

svc = svm.SVC(kernel='linear')

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

group1 = '0:2'
group2 = '0:3:2'
group3 = '0::3'
group4 = '1:3'
group5 = '1::2'
group6 = '2:'
groups = [group1, group2, group3, group4, group5, group6]


def do_both(X,y):
    svc.fit(X, y)
    plot_estimator(svc, X, y)


def plot_all_groups(data, y):
    X = data[:, 0:2]
    do_both(X, y)

    X = data[:, 0:3:2]
    do_both(X, y)

    X = data[:, 0::3]
    do_both(X, y)

    X = data[:, 1:3]
    do_both(X, y)

    X = data[:, 1::2]
    do_both(X, y)

    X = data[:, 2:]
    do_both(X, y)

# Setosa vs Versicolor
y = iris.target[0:100]
data = iris.data[0:100]
plot_all_groups(data, y)

# Versicolor vs Virginica
y = iris.target[50:]
data = iris.data[50:]
plot_all_groups(data, y)

# Setosa vs Virginica
y = np.concatenate((iris.target[:50],iris.target[100:]))
data = np.concatenate((iris.data[:50],iris.data[100:]))
plot_all_groups(data, y)

"""
Setosa vs Versicolor is completely clean for all groups.
Next uses all 3 flowers in SVM
"""
# All 3 flowers
y = iris.target
data = iris.data
plot_all_groups(data, y)

"""
Versicolor vs Virginica have no clean split for any groups.
Will use this flower comparison moving forward with wider soft margins.
"""
# Versicolor vs Virginica
y = iris.target[50:]
data = iris.data[50:]

svc = svm.SVC(kernel='linear', C=1)
plot_all_groups(data, y)

svc = svm.SVC(kernel='linear', C=2)
plot_all_groups(data, y)

svc = svm.SVC(kernel='linear', C=3)
plot_all_groups(data, y)

svc = svm.SVC(kernel='linear', C=100)
plot_all_groups(data, y)

svc = svm.SVC(kernel='linear', C=1000)
plot_all_groups(data, y)

"""I notice very small differences with wider soft margins"""

# All 3 flowers
y = iris.target
data = iris.data

svc = svm.SVC(kernel='linear', C=1)
plot_all_groups(data, y)

svc = svm.SVC(kernel='linear', C=2)
plot_all_groups(data, y)

svc = svm.SVC(kernel='linear', C=3)
plot_all_groups(data, y)

svc = svm.SVC(kernel='linear', C=100)
plot_all_groups(data, y)

svc = svm.SVC(kernel='linear', C=1000)
plot_all_groups(data, y)
