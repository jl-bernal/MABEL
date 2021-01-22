'''
Set of functions to compute the likelihoods and posteriors 
needed for the MCMC samplers
2021 - Jose Luis Bernal & Chih-Fan Chen
'''

import numpy as np
from scipy.linalg import solve_triangular
from scipy.interpolate import interp1d

from source.cosmo import H_z, get_comov_dist, get_angular_dist_z1z2
from source.utilities import gaussian,assign_params

####################################
## Flat priors for all parameters ##
####################################

def ln_flat_prior(theta,prior_min,prior_max):
    '''
    Applies flat priors to all parameters, with hard limits given by
    prior_min and prior_max
    '''
    if (prior_min <= theta).all() == True and (theta <= prior_max).all() == True:
        return 0.0
    else:
        return -np.inf
        
############################################
## Posterior call to prior and likelihood ##
############################################

def lnpost(theta,self):
    '''
    Function to call to priors and likelihoods, and pass the log_posterior
    to the MCMC sampler
    '''
    lp = ln_flat_prior(theta, self.prior_min, self.prior_max)
    if not np.isfinite(lp):
        return -np.inf
    llkl = ln_lkls(theta, self)

    if not np.isfinite(llkl):
        return -np.inf
    else:
        return lp + llkl
        
#################
## Likelihoods ##
#################

def ln_lkls(theta,self):
    '''
    Function to call and compute all the individual likelihoods included
    in the analysis
    '''
    #First assign the parameters
    params = assign_params(self,theta)
    if type(params) != dict:
        return -np.inf
    
    # Get comoving distances
    DM = get_comov_dist(self,params)
    
    chi2 = 0.
    
    # Compute chi2 for each likelihood
    if self.lkls['H0prior']:
        chi2 += chi2_prior(self.data['H0prior'],self.cov['H0prior'],params['H0'])
        
    if self.lkls['rdprior']:
        chi2 += chi2_prior(self.data['rs'],self.cov['rs'],params['rs'])
        
    if self.lkls['BAO']:
        chi2 += chi2_BAO(self.data['BAO'],self.cov['BAO'],self,DM,params)
        
    if self.lkls['SN']:
        chi2 += chi2_SN(self.data['SN'],self.cov['SN'],DM,params)

    if self.lkls['mock_highz_StrongLens_IFU']:
        chi2 += chi2_mock_highz_StrongLens_IFU(self.data['mock_highz_StrongLens_IFU'],self.cov['mock_highz_StrongLens_IFU'],DM,params)
        
    if self.lkls['Clocks']:
        chi2 += chi2_clocks(self.data['Clocks'],self.cov['Clocks'],self,params)
        
    #Return the total log likelihood
    return -0.5 * float(chi2)
    

#################
## LIKELIHOODS ##
#################

# General prior
#-----------
def chi2_prior(data,cov,param):
    '''
    Chi2 given by one or more priosr in any of the parameters. 
    Assuming Gaussian errors.
    Assuming independent measurements for all
    '''
    return np.sum(((param - data)/cov)**2)

# BAO
#-----
def chi2_BAO(data,cov,self,DM,params):
    '''
    Chi2 given by BAO-only measurements (Alcock-Paczynski effect). 
    For the moment, includes 6dF, SDSS MGS, BOSS, eBOOS and WiggleZ
    '''
    chi2 = 0.
    c_vel = 299792.458
    rs = params['rs']
    #6dF
    z = data[0][0]
    q = data[0][1]
    theo_DV = (DM(z)**2.*z*c_vel/(H_z(self,params,z)))**(1./3)
    chi2 += ((rs/theo_DV - q)/cov[0])**2.
    #MGS
    z = data[1][0]
    q = data[1][1]
    theo_DV = (DM(z)**2.*z*c_vel/(H_z(self,params,z)))**(1./3)
    chi2 += ((theo_DV/rs - q)/cov[1])**2.
    #Wiggle Z
    z = data[2][0]
    q = data[2][1]
    theo_vec = (DM(z)**2.*z*c_vel/(H_z(self,params,z)))**(1./3)
    chi2 += np.dot(theo_vec/rs-q,np.dot(cov[2],theo_vec/rs-q))
    #BOSS DR12 #only first two bins (already cut in data file)
    z = data[3][0]
    q = data[3][1]
    theo_vec = np.zeros(len(q))
    theo_vec[::2] = DM(z[::2])/rs
    theo_vec[1::2] = c_vel/(H_z(self,params,z[1::2])*rs)
    chi2 += np.dot(theo_vec-q,np.dot(cov[3],theo_vec-q))
    #eBOSS LRG
    z = data[4][0]
    q = data[4][1]
    theo_vec = np.zeros(len(q))
    theo_vec[0] = DM(z[0])/rs
    theo_vec[1] = c_vel/(H_z(self,params,z[1])*rs)
    chi2 += np.dot(theo_vec-q,np.dot(cov[4],theo_vec-q))
    #eBOSS ELGS
    z = data[5][0]
    table = data[5][1]
    theo_DV_over_rs = (DM(z)**2.*z*c_vel/(H_z(self,params,z)))**(1./3)/rs
    try:
        chi2 += -2 * np.log(table(theo_DV_over_rs))
    except:
        return np.inf
    #eBOSS QSO
    z = data[6][0]
    q = data[6][1]
    theo_vec = np.zeros(len(q))
    theo_vec[0] = DM(z[0])/rs
    theo_vec[1] = c_vel/(H_z(self,params,z[1])*rs)
    chi2 += np.dot(theo_vec-q,np.dot(cov[6],theo_vec-q))
    #eBOSS Lya auto
    z = data[7][0]
    table = data[7][1]
    theo_vec_DM_over_rs = DM(z)/rs
    theo_vec_DH_over_rs = c_vel/(H_z(self,params,z)*rs)
    try:
        chi2 += -2 * np.log(table(theo_vec_DM_over_rs,theo_vec_DH_over_rs))
    except:
        return np.inf
    #eBOSS Lya x QSO
    z = data[8][0]
    table = data[8][1]
    theo_vec_DM_over_rs = DM(z)/rs
    theo_vec_DH_over_rs = c_vel/(H_z(self,params,z)*rs)
    try:
        chi2 += -2 * np.log(table(theo_vec_DM_over_rs,theo_vec_DH_over_rs))
    except:
        return np.inf
    
    return chi2

# SNeIa
#--------
def chi2_SN(data,cov,DM,params):
    '''
    Chi2 given by SNeIa measurements from Pantheon
    '''
    z = data[:,0]
    q = data[:,1]
    M = params['M']
    moduli = DM(z)*(1.+z)
    
    chi2 = 0.
    if not all(i > 0 for i in moduli):
        return np.inf
    else:
        moduli = 5.*np.log10(moduli)+25.
        residuals = q - (M+moduli)

        residuals = solve_triangular(cov,residuals,lower=True,
                                     check_finite=False)
        return (residuals**2.).sum()
        

# mock_highz_StrongLens_IFU
#----------------------------
def chi2_mock_highz_StrongLens_IFU(data,cov,DM,params):
    '''
    Chi2 from high z strong lenses with IFU
    Assuming independent measurements!
    '''
    z_d, z_s = data[0].T[0], data[0].T[1]
    theo_vec_Dds = get_angular_dist_z1z2(DM, params, z_d, z_s)
    theo_vec_Ds = DM(z_s) / (1 + z_s)
    DsDds = theo_vec_Ds / theo_vec_Dds
    return np.sum(((DsDds - data[1])/cov)**2)
    

# Clocks
#---------
def chi2_clocks(data,cov,self,params):
    '''
    Chi2 from Cosmic Clocks H(z) measurements
    Assuming independent measurements and Gaussian errors
    '''
    z,Hzdata = data
    Hz = H_z(self,params,z)
    return np.sum(((Hz - Hzdata)/cov)**2)


