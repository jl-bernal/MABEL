'''
Set of functions to read and process data
2021 - Jose Luis Bernal & Chih-Fan Chen
'''

import numpy as np
from numpy.linalg import inv
from numpy import dot
from scipy.linalg import cholesky
from scipy.interpolate import interp1d, interp2d
from scipy import stats

from pandas import read_table
import numexpr as ne
import os


#Read the data sets fed by which data and process them as needed
def read_and_process_data(lkls):
    '''
    Reads and processes the data to be included in the MCMC
    '''
    data = dict(H0prior=None,BAO=None,SN=None,mock_highz_StrongLens_IFU=None)
    cov = dict(H0prior=None,BAO=None,SN=None,mock_highz_StrongLens_IFU=None) 
    data_path = os.path.dirname(os.path.abspath(__file__))+'/../data'

    if lkls['H0prior']:
        datafile = np.loadtxt(data_path+'/H0prior/H0prior.txt')
        try:
            data['H0prior'] = datafile[:,0]
            cov['H0prior'] = datafile[:,1]
        except:
            data['H0prior'] = datafile[0]
            cov['H0prior'] = datafile[1]
            
    if lkls['rdprior']:
        datafile = np.loadtxt(data_path+'/H0prior/H0prior.txt')
        try:
            data['rdprior'] = datafile[:,0]
            cov['rdprior'] = datafile[:,1]
        except:
            data['rdprior'] = datafile[0]
            cov['rdprior'] = datafile[1]

    if lkls['BAO']:
        dataBAO = [] #z,q
        covBAO = [] #dq

        #6dF, SDSS DR7 MGS
        #read the BAO file [z, q, dq, type]
        datafile = np.loadtxt(data_path+'/BAO/bao_no_boss_eboss.txt')
        tt = datafile[:,3]
        
        ind = np.where(tt==0)[0]
        dataBAO.append([datafile[ind,0],datafile[ind,1]])
        covBAO.append(datafile[ind,2])

        ind = np.where(tt==1)[0]
        dataBAO.append([datafile[ind,0],datafile[ind,1]])
        covBAO.append(datafile[ind,2])

        #BOSS DR12
        datafile = np.loadtxt(data_path+'/BAO/boss_dr12_bao_only_2lowz.txt')
        dataBAO.append([datafile[:,0],datafile[:,1]])
        covBAO.append(inv(np.loadtxt(data_path+'/BAO/boss_dr12_bao_only_2lowz_covmat.txt')))

        #eBOSS DR16 LRGs
        datafile = np.loadtxt(data_path+'/BAO/eboss_dr16_lrg_bao_only.txt')
        dataBAO.append([datafile[:,0],datafile[:,1]])
        covBAO.append(inv(np.loadtxt(data_path+'/BAO/eboss_dr16_lrg_bao_only_covmat.txt')))
        
        #eBOSS DR16 ELGs
        datafile = np.loadtxt(data_path+'/BAO/eboss_dr16_elg_bao_only_DVtable.txt')
        table = interp1d(datafile[:,0],datafile[:,1],bounds_error=True)
        zELG = 0.845
        dataBAO.append([zELG,table])
        covBAO.append(None)
        #eBOSS DR16 QSOs
        datafile = np.loadtxt(data_path+'/BAO/eboss_dr16_qso_bao_only.txt')
        dataBAO.append([datafile[:,0],datafile[:,1]])
        covBAO.append(inv(np.loadtxt(data_path+'/BAO/eboss_dr16_qso_bao_only_covmat.txt')))
        #eBOSS DR16 Lya auto
        datafile = np.loadtxt(data_path+'/BAO/eboss_dr16_Lya_auto_bao_only_DMDHtable.txt')
        DMrsgrid = np.unique(datafile[:,0])
        DHrsgrid = np.unique(datafile[:,1])
        lkl_ratio = datafile[:,2].reshape(50,50)
        table = interp2d(DMrsgrid,DHrsgrid,lkl_ratio.T,bounds_error=True)
        zLya = 2.334
        dataBAO.append([zLya,table])
        covBAO.append(None)
        #eBOSS DR16 Lya cross QSO
        datafile = np.loadtxt(data_path+'/BAO/eboss_dr16_Lya_x_qso_bao_only_DMDHtable.txt')
        DMrsgrid = np.unique(datafile[:,0])
        DHrsgrid = np.unique(datafile[:,1])
        lkl_ratio = datafile[:,2].reshape(50,50)
        table = interp2d(DMrsgrid,DHrsgrid,lkl_ratio.T,bounds_error=True)
        zLya = 2.334
        dataBAO.append([zLya,table])
        covBAO.append(None)
        
        data['BAO'] = dataBAO
        cov['BAO'] = covBAO
        
    if lkls['SN']:
        with open(data_path+'/SNeIa/Pantheon_lcparam_full_long.txt','r') as text:
            clean_first_line = text.readline()[1:].strip()
            names = [e.strip().replace('3rd', 'third')
                         for e in clean_first_line.split()]
        lc_parameters = read_table(data_path+'/SNeIa/Pantheon_lcparam_full_long.txt',
                sep=' ', names=names, header=0, index_col=False)
        redshifts = lc_parameters.zcmb
        mb = lc_parameters.mb
        datamat = np.zeros((redshifts.size,2))
        datamat[:,0] = redshifts
        datamat[:,1] = mb
        data['SN'] = datamat
        
        with open(data_path+'/SNeIa/Pantheon_sys_full_long.txt', 'r') as text:
            length = int(text.readline())
        SN_covmat = read_table(data_path+'/SNeIa/Pantheon_sys_full_long.txt').to_numpy().reshape((length, length))
        SN_covmat = ne.evaluate("SN_covmat")
        SN_covmat += np.diag(lc_parameters.dmb**2)

        SN_covmat = cholesky(SN_covmat, lower=True, overwrite_a=True)
        cov['SN'] = SN_covmat
        
    if lkls['Clocks']:
        datafile = np.loadtxt(data_path+'/Clocks/Clocks.txt')
        z = datafile[:,0]
        q = datafile[:,1]
        data['Clocks'] = [z,q]
        cov['Clocks'] = datafile[:,2]

    return data,cov
