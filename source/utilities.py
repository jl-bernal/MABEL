'''
Set of functions useful in some modules
2021 - Jose Luis Bernal & Chih-Fan Chen
'''

import inspect
import numpy as np
from numpy import dot
from numpy.linalg import inv
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import deepdish as dd

# Prepare matrices for spline reconstruction
def spline_preparation(z_knots):
    '''
    Prepares the matrices needed for the spline interpolation
    '''
    # Define matrices and intervals for the spline reconstruction
    # h. h has a size of n-1: h[i] = x[i+1] - x[i]
    n_mat = len(z_knots)
    h = np.zeros(n_mat - 1)
    for i in range(0, n_mat - 1):
        h[i] = z_knots[i + 1] - z_knots[i]

    # Define matrices R [(n-2)x(n-2)] and Q[(n-2)xn]
    R = np.zeros([n_mat - 2, n_mat - 2])
    Q = np.zeros([n_mat, n_mat - 2])

    for i in range(0, n_mat - 2):
        R[i, i] = 2. / 3. * (h[i] + h[i + 1])
        Q[i, i] = 1. / h[i]
        Q[i + 1, i] = - (1. / h[i] + 1. / h[i + 1])
        Q[i + 2, i] = 1. / h[i + 1]
        if (i < (n_mat - 3)):
            R[i, i + 1] = h[i + 1] / 3.
            R[i + 1, i] = h[i + 1] / 3.
    return h, R, Q
    
    
    
# Obtain the spline reconstruction from the nodes y and spacing h
def spline_reconstruction(y, h, R, Q):
    '''
    Computes the coefficients of the spline reconstruction from the 
    value of H(z) at the z_knots
    '''
    # We assume a value for the smoothing parameter:
    l = 0.  # No smoothing

    A = dot(Q.T, Q) * l + R
    bi = dot(dot(inv(A), Q.T), y)

    b = np.zeros(len(y))
    a = np.zeros(len(y))
    c = np.zeros(len(y))

    B = dot(Q, bi)
    d = y - l * B

    for p in range(1, len(y) - 1):
        b[p] = bi[p - 1]

    for p in range(0, len(y) - 1):
        a[p] = (b[p + 1] - b[p]) / 3. / h[p]
        c[p] = (d[p + 1] - d[p]) / h[p] - 1. / 3. * (b[p + 1] + 2. * b[p]) * h[p]
    return a, b, c, d
    

def get_default_params(func):
    '''
    Gets the default parameters of a function or class. Output
    is a dictionary of parameter names and values, removing any 
    potential instance of "self"
    '''
    
    args = inspect.getargspec(func)
    
    param_names = args.args
    if 'self' in param_names:
        param_names.remove('self')
    
    default_values = args.defaults
    
    default_params = dict(zip(param_names,default_values))

    return default_params



def check_params(input_params, default_params):
    '''
    Check input parameter values to ensure that they have the required type
    '''
    
    for key in input_params.keys():
        # Check if input is a valid parameter
        if key not in default_params.keys():
            raise AttributeError(key+" is not a valid parameter")
        
        input_value = input_params[key]
        default_value = default_params[key]
        
        # Check if input has the correct type
        if type(input_value)!=type(default_value):
            raise TypeError("Parameter "+key+" must be a "+
                                str(type(default_value)))
                                    
        # Special requirements for some parameters
        if key == 'sampler':
            if input_params[key] != 'zeus' and input_params[key] != 'emcee':
                raise AttributeError('Please, choose a sampler between zeus and emcee')


def check_lkls(lkls):
    '''
    Check that incompatible likelihoods are not included at the same time
    '''
    #Add here all the rules required
    if lkls['rdprior'] and not lkls['BAO']:
        raise AttributeError('Please use only a prior in rd ("rdprior" likelihood) when also using the "BAO" likelihood.')
        
def check_zmax(self):
    '''
    Check that zmax cover the maximum redshift of the data
    '''
    if self.lkls['mock_highz_StrongLens_IFU']:
        if np.max(self.data['mock_highz_StrongLens_IFU'][0].T[1] > self.zmax):
            raise ValueError('You have input zmax = {}, but mock_highz_StrongLens_IFU requires zmax >= {}.'.format(self.zmax,np.max(self.data['mock_highz_StrongLens_IFU'][0].T[1])))
        if np.max(self.data['mock_highz_StrongLens_IFU'][0].T[1] > self.z_knots[-1]) and self.expansion == 'spline':
            raise ValueError('You have input your last z_knot at z = {}, but mock_highz_StrongLens_IFU requires zmax >= {}.'.format(self.zmax,np.max(self.data['mock_highz_StrongLens_IFU'][0].T[1])))
    if self.lkls['BAO']:
        if self.zmax < self.data['BAO'][-1][0]:
            raise ValueError('You have input zmax = {}, but BAO requires zmax >= {}.'.format(self.zmax,self.data['BAO'][-1][0]))
    if self.lkls['SN']:
        if self.zmax < np.max(self.data['SN'][:,0]):
            raise ValueError('You have input zmax = {}, but SN requires zmax >= {}.'.format(self.zmax,np.max(self.data['SN'][:,0])))
    if self.lkls['Clocks']:
        if self.zmax < np.max(self.data['Clocks'][0]):
            raise ValueError('You have input zmax = {}, but Clocks require zmax >= {}.'.format(self.zmax,np.max(self.data['Clocks'][0])))
        
        
def gaussian(x,mu,sigma):
    '''
    Returns a Gaussian PDF P(x) with mean mu and std sigma
    '''
    return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    
    
def assign_params(self, theta):
    '''
    Assign the parameters at each point in the MCMC (e.g., theta) to a 
    params dictionary (and updates self.z_knot for flexknot).
    
    To be used while running
    '''
    params = {}
    if self.expansion == 'flexknot':
        self.z_knots = np.zeros(self.Nknots)
        self.z_knots[-1] = self.zmax
        self.z_knots[1:-1] = theta[:self.Nknots - 2]
        y = theta[self.Nknots-2:2*self.Nknots-2]
        H0 = y[0]
        #check all knots in order
        if np.any(np.diff(self.z_knots) <= 0.):
            return -np.inf
        else:
            # Prepare spline matrices:
            h, R, Q = spline_preparation(self.z_knots)
            a,b,c,d = spline_reconstruction(y,h,R,Q)
            params['coeffs'] = a,b,c,d
            params['H0'] = H0
    elif self.expansion == 'spline':
        y = theta[:self.Nknots]
        H0 = y[0]
        a,b,c,d = spline_reconstruction(y,self.h,self.R,self.Q)
        params['coeffs'] = a,b,c,d
        params['H0'] = H0
    else:
        params['H0'] = theta[0]
        params['Omega_m'] = theta[1]
        if self.expansion == "flatLCDM":
            pass
        elif self.expansion == "flatwCDM":
            params['w0'] = theta[2]
        elif self.expansion == "flatw0waCDM":
            params['w0'] = theta[2]
            params['wa'] = theta[3]
        elif self.expansion == "LCDM":
            pass
        elif self.expansion == "wCDM":
            params['w0'] = theta[2]
        elif self.expansion == "w0waCDM":
            params['w0'] = theta[2]
            params['wa'] = theta[3]
    #Assign the nuisance and additional parameteres
    counter = -1
    if self.lkls['SN']:
        params['M'] = theta[counter]
        counter += -1
    if self.lkls['BAO']:
        params['rs'] = theta[counter]
        counter += -1
    if not self.flat or not 'flat' in self.expansion:
        params['Omega_k'] = theta[counter]
    else:
        params['Omega_k'] = 0.
    return params


def open_MCMC(path):
    '''
    Returns the chain, the log_prob(including prior), and the information of the MCMC
    '''
    d = dd.io.load(path)
    return d['samples'],d['log_prob_samples'],d['Summary']
    
    
def find_max_mean_CLregions(samples,bandwidth=0,CL=0.6829,
                            ranges=None,printing=False,visual_check=False):
    '''
    Computes the maximum, mean and limits at a given CL for a 1D marginalized posterior. 
    
    input parameters:
    
        -samples:       MCMC samples for the parameter of interest
        
        -bandwidth:     value for the Gaussian smoothing of the histogram
                        (default: 0, no smoothing)
        
        -CL:            Confidence level (over 1) at which compute the errors.
                        (default: 0.6829, 1sigma)
                        
        -ranges:        cuts in 1D posterior before computing everything
                        (default: None)
                        
        -printing:      whether you want the results printed, 
                        if False they're returned as output of the function
                        (default: False)
    
        -visual_check: Check visually whether the bandwitdh is suitable or not.
                        (default: False)
    '''
    if not ranges:
        vmin,vmax = np.min(samples),np.max(samples)
    else:
        vmin,vmax = ranges
        
    dat = np.linspace(vmin,vmax,1000)
    dist = gaussian_kde(samples,bandwidth)(dat)
    dist *= 1./np.trapz(dist,dat)
    if visual_check:
        plt.hist(samples,bins=100,density=True,range=[vmin,vmax])
        plt.plot(dat,dist)
        plt.show()
    #Get the maximum:
    maxi = dat[np.argmax(dist)]
    #Get the mean:
    mean = np.trapz(dat*dist,dat)
    #Get the CL limits
    lim_up = np.max(dist)
    lim_down = 0
    eps = 0.001
    for i in range(0,512):
        lim = (lim_up+lim_down) / 2
        ind = np.where(dist >= lim)
        dens = np.trapz(dist[ind],dat[ind])
        if dens >= CL+eps:
            lim_down = lim
        elif dens <= CL-eps:
            lim_up = lim
        else:
            break
    low, high = dat[ind[0][0]],dat[ind[0][-1]]
    if printing:
        print('maxi = {}, mean = {}, 1sigma_low = {}, 1sigma_high = {}'.format(maxi,mean,maxi-low,high-maxi))
        return
    else:
        return maxi,mean,maxi-low,high-maxi

