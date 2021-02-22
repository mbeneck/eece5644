# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 09:58:07 2021

@author: eckmb
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

def getConfusionMatrix(classifierresults, testclasses, classes):
    classifier_results_bool = np.equal(np.transpose(np.tile(classes, (classifierresults.size, 1))), classifierresults)
    classmatrix = np.equal(np.transpose(np.tile(classes, (classifierresults.size,1))), testclasses)
    confusion_matrix = np.empty((classes.size, classes.size))

    for i in range(classes.size):
        tempclass = np.tile(classmatrix[i], (classes.size, 1))

        outvector = np.sum(np.logical_and(classifier_results_bool, tempclass), axis=1)/classifierresults.size
        confusion_matrix[:,i] = outvector
        
    return np.transpose(confusion_matrix)

def minErrorClassifier(testdata, testlabels):
    classes = np.unique(testlabels)
    classmask = np.equal(np.tile(classes, (testlabels.size,1)), np.transpose(testlabels))
    databyclass = np.empty(classes.size, dtype='object')
    covbyclass = np.empty(classes.size, dtype='object')
    regcovbyclass = np.empty(classes.size, dtype='object')
    meanbyclass = np.empty(classes.size, dtype='object')
    regularizinglambdas = np.empty(classes.size)
    classpriors = np.sum(classmask, axis = 0)/testlabels.size
    print(classpriors)
    classprobabilitymatrix = np.empty((testlabels.size, classes.size))
    
    for i in range(classes.size):
        databyclass[i] = testdata[classmask[:,i]]
        covbyclass[i] = np.cov(databyclass[i], rowvar=False)
        regularizinglambdas[i] = np.trace(covbyclass[i])/np.linalg.matrix_rank(covbyclass[i])
        print(classes[i])
        print(regularizinglambdas[i])
    
    regularizinglambda = np.average(regularizinglambdas)
    # regularizinglambda = .05
    
    print("Regularizing lambda is %f" % (regularizinglambda))
        
    for i in range(classes.size):
        regcovbyclass[i] = covbyclass[i] + regularizinglambda*np.identity(np.shape(testdata)[1])
        meanbyclass[i] = np.mean(databyclass[i], axis=0)
        classprobabilitymatrix[:,i] = classpriors[i]*multivariate_normal.pdf(testdata, meanbyclass[i], regcovbyclass[i])
        
    classifierresults = classes[np.argmax(classprobabilitymatrix, axis=1)]
    return classifierresults, covbyclass, meanbyclass, databyclass

def getErrorProbability(actual_labels, classifierresults):
        return np.sum(np.logical_not(np.equal(classifierresults, actual_labels)))/actual_labels.size*100

    

# HAR dataset

HARdata = np.loadtxt(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW1\UCI HAR Dataset\test\X_test.txt')
HARlabels = np.array([np.loadtxt(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW1\UCI HAR Dataset\test\y_test.txt', dtype='int8')])
HARclasses = np.array([1,2,3,4,5,6])

HARclassifierresults, HARcov, HARmean, HARdata = minErrorClassifier(HARdata, HARlabels)
HARconfusion = pd.DataFrame(getConfusionMatrix(HARclassifierresults, HARlabels, HARclasses), index=HARclasses, columns=HARclasses)
HARerrorprob = getErrorProbability(HARlabels, HARclassifierresults)

# Wine Quality

#only using white wine dataset
winetestdata = np.loadtxt(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW1\winequality-white.csv', delimiter=';', skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10))
winetestlabels = np.array([np.loadtxt(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW1\winequality-white.csv', delimiter=';', skiprows=1, usecols=11, dtype='int8')])
wineclasses = np.unique(winetestlabels)

wineclassifierresults, winecov, winemean, winedatabyclass = minErrorClassifier(winetestdata, winetestlabels)
wineconfusion = pd.DataFrame(getConfusionMatrix(wineclassifierresults, winetestlabels, wineclasses), index=wineclasses, columns=wineclasses)
wineerrorprob = getErrorProbability(winetestlabels, wineclassifierresults)

winemins = np.amin(winetestdata, axis=0)
winemaxs = np.amax(winetestdata, axis=0)
winenormterms = 1/(winemaxs-winemins)
winetestdata_norm = np.matmul(np.subtract(winetestdata,np.transpose(winemins)), np.diag(winenormterms))

# Analysis

wineclassifierresults_norm, winecov_norm, winemean_norm, winedatabyclass_norm = minErrorClassifier(winetestdata_norm, winetestlabels)
wineconfusion_norm = pd.DataFrame(getConfusionMatrix(wineclassifierresults_norm, winetestlabels, wineclasses), columns=wineclasses, index=wineclasses)
wineerrorprob_norm = getErrorProbability(winetestlabels, wineclassifierresults_norm)

fig, axes = plt.subplots(3,4)

for i in range(np.shape(winetestdata)[1]):
    axes.flat[i].hist(winedatabyclass[3][i])
    axes.flat[i].set_title("Feature %i" % i, fontsize=8)
    axes.flat[i].xaxis.set_ticklabels([])
    axes.flat[i].yaxis.set_ticklabels([])

axes.flat[11].set_visible(False)
fig.suptitle("Histograms for Features 0-10 in Class 6 Wines")

fig2, axes2 = plt.subplots(3,4)


xfeat = 2

for j in range(np.shape(winetestdata)[1]):
    fig2, axes2 = plt.subplots(3,4)
    
    handles = []
    
    for i in range(np.shape(winetestdata)[1]):
        handles.append(axes2.flat[i].scatter(winedatabyclass[3][:,j], winedatabyclass[3][:,i], c='r', s=.05))
        #axes2.flat[i].scatter(winedatabyclass[4][:,j], winedatabyclass[4][:,i], c='g', s=.05)
        handles.append(axes2.flat[i].scatter(winedatabyclass[2][:,j], winedatabyclass[2][:,i], c='b', s=.05))
        axes2.flat[i].xaxis.set_ticklabels([])
        axes2.flat[i].yaxis.set_ticklabels([])
        axes2.flat[i].set_xlabel('%i' % (j))
        axes2.flat[i].set_ylabel('%i' % (i), rotation=0)

    fig2.tight_layout()
    axes2.flat[11].set_visible(False)
    fig2.legend([handles[0], handles[1]], labels =['Class 6', 'Class 5'], loc='lower right')


