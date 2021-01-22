'''
Set of functions to compute expansion rate and cosmological distances
for different expansion history models
2021 - Jose Luis Bernal & Chih-Fan Chen
'''

import numpy as np
from scipy.interpolate import interp1d

# function to obtain H(z) from spline:
def H_z_spline(a, b, c, d, z_knots, z_data):
    '''
    Computes H(z) for the spline
    '''
    try:
        Ndat = len(z_data)
    except:
        z_data = np.array([z_data])
        Ndat = 1
    Hz = np.zeros(Ndat)
    for i in range(0, len(z_knots) - 1):
        ind = np.where(np.logical_and(z_data >= z_knots[i], z_data < z_knots[i + 1]))
        Hz[ind] = d[i] + c[i] * (z_data[ind] - z_knots[i]) + b[i] * (z_data[ind] - z_knots[i]) ** 2 + \
                  a[i] * (z_data[ind] - z_knots[i]) ** 3
    return Hz


def H_z_std_cosmo(H0, Omega_m, z, Omega_k=0, w0=-1, wa=0):
    '''
    Computes H(z) for the standard cosmological models
    '''
    Hz = H0 * (Omega_m * (1 + z) ** 3. + Omega_k * (1 + z) ** 2 +
                         (1 - Omega_m - Omega_k - 5.4186690102496706e-05) * (1 + z) ** (
                                     3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z)) +
                         5.4186690102496706e-05 * (1 + z) ** 4) ** 0.5
    return Hz


def H_z(self,params,z):
    '''
    Computes the H(z) for any of the available expansion history models
    '''
    if self.expansion == "flatLCDM":
        Hz = H_z_std_cosmo(params['H0'], params['Omega_m'], z)
    elif self.expansion == 'flatwCDM':
        Hz = H_z_std_cosmo(params['H0'], params['Omega_m'], z, w0=params['w0'])
    elif self.expansion == 'flatw0waCDM':
        Hz = H_z_std_cosmo(params['H0'], params['Omega_m'], z, w0=params['w0'], wa=params['wa'])
    elif self.expansion == 'LCDM':
        Hz = H_z_std_cosmo(params['H0'], params['Omega_m'], z, Omega_k=params['Omega_k'])
    elif self.expansion == 'wCDM':
        Hz = H_z_std_cosmo(params['H0'], params['Omega_m'], z, Omega_k=params['Omega_k'], w0=params['w0'])
    elif self.expansion == 'w0waCDM':
        Hz = H_z_std_cosmo(params['H0'],params['Omega_m'], z, Omega_k=params['Omega_k'],w0=params['w0'],wa=params['wa'])
    elif self.expansion == "spline" or self.expansion == "flexknot":
        a,b,c,d = params['coeffs']
        Hz = H_z_spline(a, b, c, d, self.z_knots, z)
    else:
        raise ValueError('Check your input for expansion!')
    return Hz
    
    
def get_comov_dist(self,params):
    '''
    Computes the comoving distance as function of z (returns an interpolated object)
    for any available model
    '''
    dz = 0.0001
    z_int = np.linspace(0., self.zmax, int(self.zmax / dz) + 1)
    Hz = H_z(self,params,z_int)
                  
    dz_over_H = 1. / Hz * dz
    Dc = 299792.458 * np.cumsum(dz_over_H)
    DH = 299792.458 / params['H0']

    if params['Omega_k'] == 0.:
        return interp1d(z_int, Dc, kind='linear', bounds_error=True)
    elif params['Omega_k'] > 0.:
        sqrtOk = params['Omega_k'] ** 0.5
        return interp1d(z_int, DH / sqrtOk * np.sinh(Dc * sqrtOk / DH), kind='linear', bounds_error=True)
    else:
        sqrtOk = np.abs(params['Omega_k']) ** 0.5
        return interp1d(z_int, DH / sqrtOk * np.sin(Dc * sqrtOk / DH), kind='linear', bounds_error=True)



def get_angular_dist_z1z2(DM, params, z1, z2):
    '''
    Computes the angular diameter distance between z1 and z2 
    for any available model
    '''
    DM2 = DM(z2)
    DM1 = DM(z1)
    DH = 299792.458 / params['H0']

    if params['Omega_k'] == 0.:
        DA12 = 1 / (1 + z2) * (DM2 - DM1)
    else:
        DA12 = 1 / (1 + z2) * (DM2 * (1 + params['Omega_k'] * DM1 ** 2 / DH ** 2) - DM1 * (1 + params['Omega_k'] * DM2 ** 2 / DH ** 2))
    return DA12
    

