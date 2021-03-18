# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:38:51 2021

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

# Part A

m1 = np.array([0,0,0])
C1 = np.diag([1,1,1])
p1 = .3
m2 = np.array([2, 2, 1])
C2 = np.diag([2,2,2])
p2 = .3
m3 = np.array([-3, -2, -2])
C3 = np.diag([4,4,4])
m4 = np.array([-3, 1, -5])
C4 = np.diag([9,9,9])
p3= .4

problem2classlist = np.array([1,2,3])

# some test code for printing spheres to make sure the distributions were spaced well
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# def plotSphere(center, radius, ax):
#     N=50
#     stride=2
#     # ax = axes[0,0]
#     u = np.linspace(0, 2 * np.pi, N)
#     v = np.linspace(0, np.pi, N)
#     x = center[0] + np.outer(radius*np.cos(u), radius*np.sin(v))
#     y = center[1] + np.outer(radius*np.sin(u), radius*np.sin(v))
#     z = center[2] + np.outer(np.ones(np.size(u)), radius*np.cos(v))
#     ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride)
#     ax.set_title('{0}x{0} data points, stride={1}'.format(N,stride))

# plotSphere(m1, 2, ax)
# plotSphere(m2, 2*np.sqrt(2), ax)
# plotSphere(m3, 4, ax)
# plotSphere(m4, 6, ax)

# for angle in range(0, 360):
#     ax.view_init(angle, 0)
#     plt.draw()
#     plt.pause(.001)

# random=np.random.rand(10000)
# testclasses = np.zeros(10000)
# testdata = np.zeros(10000, dtype='3float32')  # 10,0000 test data points; 1d array of tuples

# for i in range(10000):
#     if(random[i] < .3):
#         testclasses[i] = 1
#         testdata[i] = np.random.multivariate_normal(m1, C1)
#     elif(random[i] >=.6):
#         testclasses[i] = 3
#         testdata[i] = .5*np.random.multivariate_normal(m3, C3) + .5*np.random.multivariate_normal(m4, C4)
#     else:
#         testclasses[i] = 2
#         testdata[i] = np.random.multivariate_normal(m2, C2)
        

problem2testdata = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW1\problem2testdata.npy')        
problem2testclasses = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW1\problem2testclasses.npy')

probabilitymatrix = np.array([p1*multivariate_normal.pdf(problem2testdata, m1, C1), p2*multivariate_normal.pdf(problem2testdata, m2, C2), p3*(.5*multivariate_normal.pdf(problem2testdata, m3, C3) + .5*multivariate_normal.pdf(problem2testdata, m4, C4))])
# class1probs = p1*multivariate_normal.pdf(problem2testdata, m1, C1)
# class2probs = p2*multivariate_normal.pdf(problem2testdata, m2, C2)
# class3probs = p3*(.5*multivariate_normal.pdf(problem2testdata, m3, C3) + .5*multivariate_normal.pdf(problem2testdata, m4, C4))
classifierresults = np.argmax(probabilitymatrix, axis=0) + 1
classifier_correctness = np.equal(problem2testclasses, classifierresults)
classifier_wrongness = np.logical_not(np.equal(problem2testclasses, classifierresults))

confusion_matrix = pd.DataFrame(columns=["D=1", 'D=2', 'D=3'], index=['L=1', 'L=2', 'L=3'])
class1 = np.equal(problem2testclasses, 1)
class2 = np.equal(problem2testclasses, 2)
class3 = np.equal(problem2testclasses, 3)
class_matrix_bool = np.array([class1, class2, class3])



for i in range(1,4):
    currentclass = np.equal(classifierresults, i)
    confusion_matrix.iat[0, i-1] = np.sum(np.logical_and(currentclass, class1))/10000
    confusion_matrix.iat[1, i-1] = np.sum(np.logical_and(currentclass, class2))/10000
    confusion_matrix.iat[2, i-1] = np.sum(np.logical_and(currentclass, class3))/10000


fig = plt.figure()
ax = fig.add_subplot(311, projection='3d')
ax.plot(problem2testdata[np.logical_and(class1, classifier_correctness)][:,0],problem2testdata[np.logical_and(class1, classifier_correctness)][:,1], problem2testdata[np.logical_and(class1, classifier_correctness)][:,2], 'go', label='Class 1 (Correct)')
ax.plot(problem2testdata[np.logical_and(class1, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class1, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class1, classifier_wrongness)][:,2], 'ro', label='Class 1 (Incorrect)')
ax.plot(problem2testdata[np.logical_and(class2, classifier_correctness)][:,0],problem2testdata[np.logical_and(class2, classifier_correctness)][:,1], problem2testdata[np.logical_and(class2, classifier_correctness)][:,2], 'g^', label='Class 2 (Correct)')
ax.plot(problem2testdata[np.logical_and(class2, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class2, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class2, classifier_wrongness)][:,2], 'r^', label='Class 2 (Incorrect)')
ax.plot(problem2testdata[np.logical_and(class3, classifier_correctness)][:,0],problem2testdata[np.logical_and(class3, classifier_correctness)][:,1], problem2testdata[np.logical_and(class3, classifier_correctness)][:,2], 'gs', label='Class 3 (Correct)')
ax.plot(problem2testdata[np.logical_and(class3, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class3, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class3, classifier_wrongness)][:,2], 'rs', label='Class 3 (Incorrect)')
ax.legend()
ax.set_title("Test Data Classification Results")
fig.show()

# Part B

lambda_10 = np.array([[0, 1, 10], [1,0,10], [1, 1, 0]])
lambda_100 = np.array([[0,1,100],[1,0,100],[1,1,0]])

riskmat_10= np.matmul(lambda_10, probabilitymatrix)
riskmat_100 = np.matmul(lambda_100, probabilitymatrix)
risk_10_classresults = np.argmin(riskmat_10, axis=0) + 1
ex_risk_10 = np.mean(np.amin(riskmat_10, axis=0))
risk_100_classresults = np.argmin(riskmat_100, axis=0) + 1
ex_risk_100 = np.mean(np.amin(riskmat_100, axis=0))

risk_10_confmatrix = getConfusionMatrix(risk_10_classresults, problem2testclasses, problem2classlist)
risk_100_confmatrix = getConfusionMatrix(risk_100_classresults, problem2testclasses, problem2classlist)

classifier_correctness = np.equal(problem2testclasses, risk_10_classresults)
classifier_wrongness = np.logical_not(np.equal(problem2testclasses, risk_10_classresults))

ax2 = fig.add_subplot(312, projection='3d')
ax2.plot(problem2testdata[np.logical_and(class1, classifier_correctness)][:,0],problem2testdata[np.logical_and(class1, classifier_correctness)][:,1], problem2testdata[np.logical_and(class1, classifier_correctness)][:,2], 'go', label='Class 1 (Correct)')
ax2.plot(problem2testdata[np.logical_and(class1, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class1, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class1, classifier_wrongness)][:,2], 'ro', label='Class 1 (Incorrect)')
ax2.plot(problem2testdata[np.logical_and(class2, classifier_correctness)][:,0],problem2testdata[np.logical_and(class2, classifier_correctness)][:,1], problem2testdata[np.logical_and(class2, classifier_correctness)][:,2], 'g^', label='Class 2 (Correct)')
ax2.plot(problem2testdata[np.logical_and(class2, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class2, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class2, classifier_wrongness)][:,2], 'r^', label='Class 2 (Incorrect)')
ax2.plot(problem2testdata[np.logical_and(class3, classifier_correctness)][:,0],problem2testdata[np.logical_and(class3, classifier_correctness)][:,1], problem2testdata[np.logical_and(class3, classifier_correctness)][:,2], 'gs', label='Class 3 (Correct)')
ax2.plot(problem2testdata[np.logical_and(class3, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class3, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class3, classifier_wrongness)][:,2], 'rs', label='Class 3 (Incorrect)')
ax2.legend()
ax2.set_title("Risk 10 Classification Results")

classifier_correctness = np.equal(problem2testclasses, risk_100_classresults)
classifier_wrongness = np.logical_not(np.equal(problem2testclasses, risk_100_classresults))

ax3 = fig.add_subplot(313, projection='3d')
ax3.plot(problem2testdata[np.logical_and(class1, classifier_correctness)][:,0],problem2testdata[np.logical_and(class1, classifier_correctness)][:,1], problem2testdata[np.logical_and(class1, classifier_correctness)][:,2], 'go', label='Class 1 (Correct)')
ax3.plot(problem2testdata[np.logical_and(class1, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class1, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class1, classifier_wrongness)][:,2], 'ro', label='Class 1 (Incorrect)')
ax3.plot(problem2testdata[np.logical_and(class2, classifier_correctness)][:,0],problem2testdata[np.logical_and(class2, classifier_correctness)][:,1], problem2testdata[np.logical_and(class2, classifier_correctness)][:,2], 'g^', label='Class 2 (Correct)')
ax3.plot(problem2testdata[np.logical_and(class2, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class2, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class2, classifier_wrongness)][:,2], 'r^', label='Class 2 (Incorrect)')
ax3.plot(problem2testdata[np.logical_and(class3, classifier_correctness)][:,0],problem2testdata[np.logical_and(class3, classifier_correctness)][:,1], problem2testdata[np.logical_and(class3, classifier_correctness)][:,2], 'gs', label='Class 3 (Correct)')
ax3.plot(problem2testdata[np.logical_and(class3, classifier_wrongness)][:,0],problem2testdata[np.logical_and(class3, classifier_wrongness)][:,1], problem2testdata[np.logical_and(class3, classifier_wrongness)][:,2], 'rs', label='Class 3 (Incorrect)')
ax3.legend()
ax3.set_title("Risk 100 Classification Results")
