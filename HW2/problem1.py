import numpy as np
import matplotlib.pyplot as plt


def hw2q1():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[0,:],data[1,:],data[2,:],'Training Dataset')
    xTrain = data[0:2,:]
    yTrain = data[2,:]
    
    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[0,:],data[1,:],data[2,:],'Validation Dataset')
    xValidate = data[0:2,:]
    yValidate = data[2,:]
    
    return xTrain,yTrain,xValidate,yValidate

def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3,.4,.3] # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x,labels = generateDataFromGMM(N,gmmParameters)
    return x

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

def plot3(a,b,c,title,mark="o",col="b"):
  from matplotlib import pyplot
  import pylab
  from mpl_toolkits.mplot3d import Axes3D
  pylab.ion()
  fig = pylab.figure()
  ax = Axes3D(fig)
  ax.scatter(b, a, c,marker=mark,color=col)
  ax.set_xlabel("x2")
  ax.set_ylabel("x1")
  ax.set_zlabel("y")
#  ax.set_aspect('equal')
  set_aspect_equal_3d(ax)
  ax.set_title(title)
  
  
def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                        for lims, mean_ in ((xlim, xmean),
                                            (ylim, ymean),
                                            (zlim, zmean))
                        for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
    
def little_phi(x):
    return np.array([1, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], x[0]**3, x[1]**3, x[0]**2*x[1], x[1]**2*x[0]])
    
def big_phi(data):
    return np.transpose(np.apply_along_axis(little_phi, 0, data))

def w_hat_ml(x, y):
    bigphi = big_phi(x)
    bigphi_transpose = np.transpose(bigphi)
    return np.linalg.inv(bigphi_transpose @ bigphi) @ bigphi_transpose @ y

def w_hat_map(x, y, gamma):
    bigphi = big_phi(x)
    bigphi_transpose = np.transpose(bigphi)
    return np.linalg.inv(1/gamma*np.eye(bigphi.shape[1])+bigphi_transpose @ bigphi) @ bigphi_transpose @ y

def w_hat_map_manygammas(x,y,gammas):
    result = np.empty((10, gammas.shape[0]))
    for i in range(gammas.shape[0]):
        result[:,i] = w_hat_map(x,y, gammas[i])
    return result

def c(x,w):
    return np.transpose(w) @ little_phi(x)

def mse(y1, y2):
    return ((y1-y2)**2).mean(axis=0)

    
# xtrain, ytrain, xvalidate, yvalidate = hw2q1()
xtrain = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p1_xtrain.npy')
ytrain = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p1_ytrain.npy')
xvalidate = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p1_xvalidate.npy')
yvalidate = np.load(r'C:\Users\eckmb\OneDrive - Northeastern University\Courses\EECE5564\HW2\p1_yvalidate.npy')


w_hat_ml_train = w_hat_ml(xtrain, ytrain)
gammas = np.power(np.array([10.0]), np.arange(-4, 5))
w_hat_map_train = w_hat_map_manygammas(xtrain, ytrain, gammas)

ml_estimatedy = np.apply_along_axis(c, 0, xvalidate, w=w_hat_ml_train)
map_estimatedy = np.transpose(np.apply_along_axis(c, 0, xvalidate, w=w_hat_map_train))

ml_mse = mse(yvalidate, ml_estimatedy)
map_mse = mse(np.transpose(np.tile(yvalidate,[gammas.shape[0], 1])), map_estimatedy)

plt.plot(gammas, map_mse, 'ro')
plt.xscale('log')
plt.title("MAP Estimator MSE vs. Gamma")
plt.xlabel("Gamma")
plt.ylabel("MSE")

