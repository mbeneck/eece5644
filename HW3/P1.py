# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:41:22 2021

@author: eckmb
"""
import numpy as np
from scipy.stats import special_ortho_group
from scipy.stats import multivariate_normal
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Problem 1

# Generate Data:

def generate_random_covariance_matrix(max_eig, min_eig, dims):
    eigenvalues = np.diag(np.random.uniform(high=max_eig, low=min_eig, size=dims))
    random_rotation = special_ortho_group.rvs(dims)
    return random_rotation @ eigenvalues @ np.transpose(random_rotation)

def classifyBayesian(data, means, covariances, priors):
    classcond_matrix = np.column_stack((multivariate_normal.pdf(data, means[:,0], covariances[:,:,0]), multivariate_normal.pdf(data, means[:,1], covariances[:,:,1]), multivariate_normal.pdf(data, means[:,2], covariances[:,:,2]), multivariate_normal.pdf(data, means[:,3], covariances[:,:,3])))
    posteriors = np.multiply(np.transpose(priors), classcond_matrix)
    return np.argmax(posteriors, axis=1)+1

def generateDataFromGMM(N,gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors'] # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0] # Data dimensionality
    C = len(priors) # Number of components
    x = np.zeros((n,N))
    labels = np.zeros((1,N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1,N))
    thresholds = np.zeros((1,C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:,l]))
        Nl = len(indl[1])
        labels[indl] = (l+1)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:,l], covMatrices[:,:,l], Nl))
        
    return x,labels



m0 = np.array([1,1,1])
m1 = np.array([1,-1,-1])
m2 = np.array([-1, -1, -1])
m3 = np.array([-1, -1, 1])

gmmParameters = {}
gmmParameters['priors'] = [.25, .25, .25, .25] # priors should be a row vector
gmmParameters['meanVectors'] = np.transpose([m0, m1, m2, m3])
gmmParameters['covMatrices'] = np.zeros((3, 3, 4))
for i in range(4):
    gmmParameters['covMatrices'][:,:,i] = generate_random_covariance_matrix(1.4, .5, 3)


train100_data, train100_labels = generateDataFromGMM(100, gmmParameters)
train200_data, train200_labels = generateDataFromGMM(200, gmmParameters)
train500_data, train500_labels = generateDataFromGMM(500, gmmParameters)
train1000_data, train1000_labels = generateDataFromGMM(1000, gmmParameters)
train2000_data, train2000_labels = generateDataFromGMM(2000, gmmParameters)
train5000_data, train5000_labels = generateDataFromGMM(5000, gmmParameters)
test100k_data, test100k_labels = generateDataFromGMM(100000, gmmParameters)

np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\gmmParameters.npy', gmmParameters, allow_pickle=True)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train100_data.npy', train100_data)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train100_labels.npy', train100_labels)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train200_data.npy', train200_data)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train200_labels.npy', train200_labels)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train500_data.npy', train500_data)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train500_labels.npy', train500_labels)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train1000_data.npy', train1000_data)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train1000_labels.npy', train1000_labels)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train2000_data.npy', train2000_data)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train2000_labels.npy', train2000_labels)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train5000_data.npy', train5000_data)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train5000_labels.npy', train5000_labels)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\test100k_data.npy', test100k_data)
np.save(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\test100k_labels.npy', test100k_labels)

#gmmParameters = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\gmmParameters.npy', allow_pickle=True)
train100_data = np.transpose(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train100_data.npy'))
train100_labels = np.ravel(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train100_labels.npy'))
train200_data = np.transpose(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train200_data.npy'))
train200_labels = np.ravel(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train200_labels.npy'))
train500_data = np.transpose(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train500_data.npy'))
train500_labels = np.ravel(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train500_labels.npy'))
train1000_data = np.transpose(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train1000_data.npy'))
train1000_labels = np.ravel(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train1000_labels.npy'))
train2000_data =np.transpose( np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train2000_data.npy'))
train2000_labels = np.ravel(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train2000_labels.npy'))
train5000_data = np.transpose(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train5000_data.npy'))
train5000_labels = np.ravel(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\train5000_labels.npy'))
test100k_data = np.transpose(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\test100k_data.npy'))
test100k_labels = np.ravel(np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW3\test100k_labels.npy'))

bayesianResults = classifyBayesian(test100k_data, gmmParameters['meanVectors'], gmmParameters['covMatrices'], gmmParameters['priors'])
bayesianErrorProb = 1- np.sum(np.equal(test100k_labels, bayesianResults))/np.size(test100k_labels)

def kfold_crossvalidate(data, labels, k, classifier):
    # classifier shold be an MLPClassifier?
    kf = KFold(n_splits=k)
    error_rates = np.zeros(k)
    iteration = 0
    for train_index, test_index in kf.split(data, labels):
        classifier.fit(data[train_index], labels[train_index])
        error = 1 - classifier.score(data[test_index], labels[test_index])
        error_rates[iteration] = error
        iteration += 1
    return error_rates, np.average(error_rates)

def select_model_order(data, labels, min_perceptrons, max_perceptrons, max_iter):
    neuron_counts = np.arange(min_perceptrons, max_perceptrons+1)
    avg_errors = np.zeros(neuron_counts.size)
    i = 0
    for neurons in neuron_counts:
        mlp = MLPClassifier(hidden_layer_sizes=neurons, max_iter=max_iter, activation="logistic")
        errvector, error = kfold_crossvalidate(data, labels, 10, mlp)
        avg_errors[i] = error
        i += 1
    return neuron_counts, avg_errors, neuron_counts[np.argmin(avg_errors)]
    # return array of number of perceptrons and error rates, as well as mininum err number of perceptrons


# # will have 1-4 neurons in hidden layer

train100_neurons, train100_errors, train100_bestcount = select_model_order(train100_data, train100_labels, 1, 4, 10000)
train200_neurons, train200_errors, train200_bestcount = select_model_order(train200_data, train200_labels, 1, 4, 10000)
train500_neurons, train500_errors, train500_bestcount = select_model_order(train500_data, train500_labels, 1, 4, 10000)
train1000_neurons, train1000_errors, train1000_bestcount = select_model_order(train1000_data, train1000_labels, 1, 4, 10000)
train2000_neurons, train2000_errors, train2000_bestcount = select_model_order(train2000_data, train2000_labels, 1, 4, 10000)
train5000_neurons, train5000_errors, train5000_bestcount = select_model_order(train5000_data, train5000_labels, 1, 4, 10000)

def get_best_trained_MLP(training_data, training_labels, hidden_layer_neurons, max_training_iterations=10000, iterations=3):
    models = list()
    model_errs = np.empty(iterations)

    for i in range(iterations):
        models.append(MLPClassifier(activation="logistic", hidden_layer_sizes=hidden_layer_neurons, max_iter=max_training_iterations))
        models[i].fit(training_data, training_labels)
        model_errs[i] = 1- models[i].score(training_data, training_labels)
    return models[np.argmin(model_errs)]

mlp100 = get_best_trained_MLP(train100_data, train100_labels, train100_bestcount)
mlp200 = get_best_trained_MLP(train200_data, train200_labels, train200_bestcount)
mlp500 = get_best_trained_MLP(train500_data, train500_labels, train500_bestcount)
mlp1000 = get_best_trained_MLP(train1000_data, train1000_labels, train1000_bestcount)
mlp2000 = get_best_trained_MLP(train2000_data, train2000_labels, train2000_bestcount)
mlp5000 = get_best_trained_MLP(train5000_data, train5000_labels, train5000_bestcount)

mlp100_err = 1- mlp100.score(test100k_data, test100k_labels)
mlp200_err = 1- mlp200.score(test100k_data, test100k_labels)
mlp500_err = 1- mlp500.score(test100k_data, test100k_labels)
mlp1000_err = 1- mlp1000.score(test100k_data, test100k_labels)
mlp2000_err = 1- mlp2000.score(test100k_data, test100k_labels)
mlp5000_err = 1- mlp5000.score(test100k_data, test100k_labels)

errs = 100*np.array([mlp100_err, mlp200_err, mlp500_err, mlp1000_err, mlp2000_err, mlp5000_err])
datasets = np.array([100, 200, 500, 1000, 2000, 5000])

plt.plot(datasets, errs, 'ro-', label="MLP")
plt.hlines(100*bayesianErrorProb, 0, 5000, 'b', label="Optimal")
plt.title("MLP Error Rate vs Optimal Bayesian Classifier")
plt.ylabel("Error Rate (%)")
plt.xlabel("Training Dataset Points")
plt.legend()
plt.xlim(0,5000)
