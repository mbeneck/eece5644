# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:51:50 2021

@author: eckmb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


p2_priors = np.array([0.65, 0.35])
p2_weights = np.array([.5, .5])
p2_class0_mean = np.array([[3,0],[0,3]])
p2_class0_cov = np.array([[[2,0],[0,1]],[[1,0],[0,2]]])
p2_class1_mean = np.array([2,2])
p2_class1_cov = np.array([[1,0],[0,1]])

def generateData(numsamples):
    classes = np.where(np.random.rand(numsamples) >= .65, 1, 0)    
    data = np.empty((numsamples,2))

    for i in range(numsamples):
        if(classes[i] == 1):
            data[i] = np.random.multivariate_normal(p2_class1_mean, p2_class1_cov)
        else:
            rand = np.random.rand()
            subclass = 1 if rand>p2_weights[0] else 0
            data[i] = np.random.multivariate_normal(p2_class0_mean[subclass], p2_class0_cov[subclass])
           
    return data, classes

def class1_condprob(x):
    return multivariate_normal.pdf(x, p2_class1_mean, p2_class1_cov)    

def class0_condprob(x):
    return p2_weights[0]*multivariate_normal.pdf(x, p2_class0_mean[0], p2_class0_cov[0]) + p2_weights[1]*multivariate_normal.pdf(x, p2_class0_mean[1], p2_class0_cov[1])
    
def classifyBayesian(data, priors):
    classcond_matrix = np.column_stack((class0_condprob(data), class1_condprob(data)))
    posteriors = np.multiply(np.transpose(priors), classcond_matrix)
    return np.argmax(posteriors, axis=1)

def classifyByLikelihoodRatio(data, gamma):
    likelihoodratios = np.divide(class1_condprob(data), class0_condprob(data))
    return np.where(likelihoodratios > gamma, 1, 0)

def calculateROCVectors(results, truelabels):
    truepoz = np.sum(np.logical_and(truelabels, results))/results.size
    falsepoz = np.sum(np.logical_and(np.logical_not(truelabels), results))/results.size
    return falsepoz, truepoz

def calculate_correct_rate(classifiedlabels, correctlabels):
    return np.sum(np.equal(classifiedlabels, correctlabels))/classifiedlabels.size*100
    
def calculate_error_rate(classifiedlabels, correctlabels):
    return 100-calculate_correct_rate(classifiedlabels, correctlabels)
    
    
        
# D20_train_data, D20_train_labels = generateData(20)
# D200_train_data, D200_train_labels = generateData(200)
# D2000_train_data, D2000_train_labels = generateData(2000)
# D10k_validate_data, D10k_validate_labels = generateData(10000)

D20_train_data = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p2_d20_data.npy')
D20_train_labels = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p2_d20_labels.npy')
D200_train_data = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p2_d200_data.npy')
D200_train_labels = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p2_d200_labels.npy')
D2000_train_data = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p2_d2000_data.npy')
D2000_train_labels = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p2_d2000_labels.npy')
D10k_validate_data = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p2_d10k_data.npy')
D10k_validate_labels = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p2_d10k_labels.npy')

# Part 1
D10k_classifier_results = classifyBayesian(D10k_validate_data, p2_priors)
testgammas = np.arange(0, 10, .01)
falsepoz = np.empty(testgammas.shape[0])
truepoz = np.empty(testgammas.shape[0])
for i in range(testgammas.shape[0]):
    classifier_results = classifyByLikelihoodRatio(D10k_validate_data, testgammas[i])
    falsepoz[i], truepoz[i] = calculateROCVectors(classifier_results, D10k_validate_labels)

plt.plot(falsepoz, truepoz, label="ROC Curve")
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title("ROC Curve for Likelihood Ratio Classifier")

optimumgamma = p2_priors[0]/p2_priors[1]
optimumresults = classifyByLikelihoodRatio(D10k_validate_data, optimumgamma)
optimumfalse, optimumtrue = calculateROCVectors(optimumresults, D10k_validate_labels)

plt.plot(optimumfalse, optimumtrue, 'ro', label="Optimum Gamma")
plt.legend()

optimumperformancerate = calculate_correct_rate(optimumresults, D10k_validate_labels)

# Part 2

# Borrowed from https://stackabuse.com/gradient-descent-in-python-implementation-and-theory/
def gradient_descent(max_iterations,threshold,w_init,
                     obj_func,grad_func,
                     learning_rate=0.05, **kwargs):
    
    w = w_init
    w_history = w
    f_history = obj_func(w,**kwargs)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    
    while  i<max_iterations and diff>threshold:
        delta_w = -learning_rate*grad_func(w,**kwargs)
        w = w+delta_w
        
        # store the history of w and f
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,obj_func(w,**kwargs)))
        
        # update iteration number and diff between successive values
        # of objective function
        i+=1
        diff = np.absolute(f_history[-1]-f_history[-2])
    
    return w_history,f_history

def sigmoid(a):
    return 1/(1+np.exp(-a))

# phi should have data columns and parameter rows for this to work
def err_func_gradient(w, **kwargs):
    phi = kwargs['phi']
    labels = kwargs['labels']
    a = w @ phi
    return np.sum(np.multiply(sigmoid(a)-labels, phi), axis = 1)

def err_func(w, **kwargs):
    phi = kwargs['phi']
    labels = kwargs['labels']
    a = w @ phi
    return -np.sum((np.multiply(labels, np.log(sigmoid(a)))+np.multiply((1-labels), np.log(1-sigmoid(a)))), axis=0)

def phi_linear(x):
    return np.vstack((np.ones(x.shape[0]), x[:,0], x[:,1]))

def phi_quadratic(x):
    return np.vstack((np.ones(x.shape[0]), x[:,0], x[:,1], x[:,0]**2, x[:,0]*x[:,1], x[:,1]**2))

def classify_logistic_linear(data, w):
    return np.where(sigmoid(w @ phi_linear(data))> .5, 1, 0)

    
def classify_logistic_quadratic(data, w):
        return np.where(sigmoid(w @ phi_quadratic(data))> .5, 1, 0)
    


d20_linear_w, d20_linear_f = gradient_descent(10000, .01, np.array([1, 0, 0]), err_func, err_func_gradient, learning_rate=.001, phi=phi_linear(D20_train_data), labels=D20_train_labels)
d200_linear_w, d200_linear_f = gradient_descent(10000, .01, np.array([1, 0, 0]), err_func, err_func_gradient, learning_rate=.001, phi=phi_linear(D200_train_data), labels=D200_train_labels)
d2000_linear_w, d2000_linear_f = gradient_descent(10000, .01, np.array([1, 0, 0]), err_func, err_func_gradient,learning_rate=.0001, phi=phi_linear(D2000_train_data), labels=D2000_train_labels)

d20_est_linear_w = d20_linear_w[-1,:]
d200_est_linear_w = d200_linear_w[-1,:]
d2000_est_linear_w = d2000_linear_w[-1,:]

d20est_linear_results = classify_logistic_linear(D10k_validate_data, d20_est_linear_w)
d200est_linear_results = classify_logistic_linear(D10k_validate_data, d200_est_linear_w)
d2000est_linear_results = classify_logistic_linear(D10k_validate_data, d2000_est_linear_w)

d20_quad_w, d20_quad_f = gradient_descent(10000, .01, np.ones(6), err_func, err_func_gradient, learning_rate=.001, phi=phi_quadratic(D20_train_data), labels=D20_train_labels)
d20_est_quad_w = d20_quad_w[-1,:]
d20est_quad_results = classify_logistic_quadratic(D10k_validate_data, d20_est_quad_w)
d200_quad_w, d200_quad_f = gradient_descent(10000, .01, np.zeros(6), err_func, err_func_gradient, learning_rate=.00001, phi=phi_quadratic(D200_train_data), labels=D200_train_labels)
d200_est_quad_w = d200_quad_w[-1,:]
d200est_quad_results = classify_logistic_quadratic(D10k_validate_data, d200_est_quad_w)
d2000_quad_w, d2000_quad_f = gradient_descent(10000, .01, np.zeros(6), err_func, err_func_gradient, learning_rate=.00001, phi=phi_quadratic(D2000_train_data), labels=D2000_train_labels)
d2000_est_quad_w = d2000_quad_w[-1,:]
d2000est_quad_results = classify_logistic_quadratic(D10k_validate_data, d2000_est_quad_w)




