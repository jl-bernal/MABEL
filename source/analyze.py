'''
Base module for analyzing the MCMCs
2021 - Jose Luis Bernal & Chih-Fan Chen
'''

#Import the needed packages
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

from source.utilities import open_MCMC,find_max_mean_CLregions,spline_preparation,spline_reconstruction
from source.cosmo import H_z,H_z_spline

class Analyze(object):
    '''
    An object controlling all relevant quantities needed to analyze the posterior
    obtained by running mabel.
    
    INPUT PARAMETERS:
    ------------------
    
        -path:      Path to the file storing the MCMC       
        
        -expansion: Type of expansion assumed in the MCMC run 
                    (choose between: 'spline','flexknot','flatLCDM','flatwCDM','flatw0waCDM','LCDM',"wCDM","w0waCDM")
                    
        -CL:        Confidence level (over 1) within to draw the samples 
                    (default: 0.6829, 1sigma)
    '''
    def __init__(self, path, expansion, CL=0.6829):
        
        #Open and extract the MCMC from the file
        self.samples_raw, self.logprob_raw,self.summary = open_MCMC(path) 
        expansion_list = ['spline','flexknot','flatLCDM','flatwCDM','flatw0waCDM','LCDM',"wCDM","w0waCDM"]
        if expansion not in expansion_list:
            print(expansion_list)
            raise ValueError('Please enter either one of types of the expansion history above')
            
        self.expansion = expansion
        if CL >= 1.:
            raise ValueError('Please remember than "CL" is defined over 1, hence 1sigma -> CL=0.6829')
        self.CL = CL
        
        
        
    def visualize_chain(self):
        '''
        Plots each walker for each parameter to estimate the burn-in phase
        '''
        ndim = self.samples_raw.shape[-1]
        plt.figure(figsize=(16,1.5*ndim))
        for n in range(ndim):
            plt.subplot2grid((ndim, 1), (n, 0))
            plt.plot(self.samples_raw[:,:,n], alpha=0.5,lw=0.8)
            plt.tick_params(axis='both',width=1,length=8,labelsize=20)
            plt.tick_params(axis='both',which='minor',width=1,length=3)
        plt.xlabel('steps',fontsize=22)
        plt.tight_layout()
        plt.show()
    
        return
    
    
    def process(self,flatten=True,thinning=1,burnin=0):
        '''
        Processes the samples_raw and the log_prob_raw, 
        with the option of thinning them, removing burn-in phase and flatten 
        the samples. 
        
        Returns processed samples and log_prob
        
        Input parameters:
        
            -flatten:   Boolean to either flat or not the samples (default: True)
        
            -thining:   Int determining by how much the samples will be thinned
                        after removing the burnin phase (default: 1, no thinning)
        
            -burnin:    Int determining the length of the burnin phase to be removed
                        (default: 0)
        '''
        ndim = self.samples_raw.shape[-1]
        if flatten:
            self.samples = self.samples_raw[burnin::thinning,:,:].reshape((-1,ndim),order='F')
            self.log_prob = self.logprob_raw[burnin::thinning,:].reshape((-1,),order='F')
        else:
            self.samples = self.samples_raw[burnin::thinning,:,:]
            self.log_prob = self.logprob_samples[burnin::thinning,:]
            
        return
        
    def print_info(self):
        '''
        Prints the info about the MCMC run
        '''
        print(self.summary['data'])
        print('Expansion model: {}'.format(self.expansion))
        if self.expansion == 'flexknot':
            print(self.summary['knots info'])
            print('Flat Universe?', self.summary['flat'])
            print(self.summary['chain stats'])
            print('auto correlation times: ', self.summary['Correlation time'])
        elif self.expansion == 'spline':
            print(self.summary['node position'])
            print('Flat Universe?', self.summary['flat'])
            print(self.summary['chain stats'])
            print('auto correlation times: ', self.summary['Correlation time'])
        else:
            print(self.summary['chain stats'])
            print('auto correlation times: ', self.summary['Correlation time'])
        return
        
    def get_1d_constraints(self,pars,bandwidth=0.,
                           ranges=None,visual_check=False):
        '''
        Computes the maximum, mean and limits at a given CL for the 1D marginalized posterior
        of the parameters required 
        
        input parameters:
        
            -pars:          List of the indices of the parameter of interest
                
            -bandwidth:     value(s) for the Gaussian smoothing of the histogram
                            (default: 0, no smoothing)
                            
            -ranges:        cuts in 1D posterior before computing everything
                            list of lists
                            (default: None)
                            
            -printing:      whether you want the results printed, 
                            if False they're returned as output of the function
                            (default: False)
        
            -visual_check: Check visually whether the bandwitdh is suitable or not.
                            (default: False)
            '''
        
        #Checks and homogeneization
        Npars = len(pars)
        if len(self.samples.shape) == 3:
            raise AttributeError('Please, flatten the samples with "process" before using "get_1d_contraints"')
        
        if type(pars) != list:
            pars = [pars]
            
        if type(bandwidth) == list:
            if len(bandwidth) != Npars:
                raise ValueError('Please input the same number of banwidths than parameters to get constraints')
        else:
            bandwidth = [bandwidth for i in range(Npars)]
            
        if ranges == None:
            ranges = [None for i in range(Npars)]
        else:
            if len(ranges) != Npars:
                raise ValueError('Please input the same number of range limits than parameters to get constraints')
                
        #Get the constraints
        for ipar in range(Npars):
            print('Parameter {} [index {}]:'.format(ipar,pars[ipar]))
            print('------------------------')
            find_max_mean_CLregions(self.samples[:,pars[ipar]],bandwidth=bandwidth[ipar],CL=self.CL,
                            ranges=ranges[ipar],printing=True,visual_check=visual_check)
            
        return
        
    def sample_chain(self,Nsamples=500):
        '''
        Get Nsamples steps from the chain within the CL  of interest from the best fit
        Returns the best fit and the Nsamples.
        
        Input parameters:
        
            -Nsamples:  Number of samples to extract (default:500)
        '''
        
        if len(self.samples.shape) == 3:
            raise AttributeError('Please, flatten the samples with "process" before using "sample_chain"')
            
        #Get indices ordered by likelihood and the best fit
        ARGSORT = np.argsort(-self.log_prob) #argsort only gets increasing order, so min(-lkl) -> max(lkl)
        ind_maxlkl = np.where(self.log_prob == np.max(self.log_prob))[0][0]
        #All samples within the CL interval with best likelihood
        NCL_samples = int(round(len(self.log_prob) * self.CL))
        #Get best fit
        self.bestfit = self.samples[ind_maxlkl,:]
        #Take the random sample
        self.samples_lim_CL = self.samples[ARGSORT[np.random.randint(0,NCL_samples,Nsamples)],:]
        
        return
        
    def get_Hz_samples(self,Nsamples=500):
        '''
        Returns the H(z) to plot.
        
        Input parameters:
        
            -Nsamples:  Number of samples to extract (default:500)
        '''
        self.sample_chain(Nsamples)
        #Get the parameters needed:
        if self.expansion == 'flexknot':
            #Get z_knots
            Nknots = int(self.summary['knots info'].split('knots')[0][0])
            zmax = float(self.summary['knots info'].split('=')[1][1:])
            #z vector
            zdat = np.linspace(0.,zmax,1000)
            #bestfit case
            bfknot = np.zeros(Nknots)
            bfknot[0] = 0.
            bfknot[-1] = zmax
            bfknot[1:-1] = self.bestfit[:Nknots-2]
            h,R,Q = spline_preparation(bfknot)
            #reconstruct spline for bestfit:
            a,b,c,d = spline_reconstruction(self.bestfit[Nknots-2:2*Nknots-2],h,R,Q)
            Hz_bestfit = H_z_spline(a,b,c,d,bfknot,zdat)
            Hz_bestfit[-1] = d[-1]
            #Samples within CL:
            Hz_CLsamples = []
            samples_knot = np.zeros(Nknots)
            samples_knot[0] = 0.
            samples_knot[-1] = zmax
            for i in range(Nsamples):
                samples_knot[1:-1] = self.samples_lim_CL[i,:Nknots-2]
                h,R,Q = spline_preparation(samples_knot)
                a,b,c,d = spline_reconstruction(self.samples_lim_CL[i,Nknots-2:2*Nknots-2],h,R,Q)
                Hz_i = H_z_spline(a,b,c,d,samples_knot,zdat)
                Hz_i[-1] = d[-1]
                Hz_CLsamples.append(Hz_i)
                
        elif self.expansion == 'spline':
            zknots = self.summary['knot position']
            Nknots = len(zknots)
            h,R,Q = spline_preparation(zknots)
            zdat = np.linspace(0,zknots[-1],1000)
            #reconstruct spline for bestfit:
            a,b,c,d = spline_reconstruction(self.bestfit[:Nknots],h,R,Q)
            Hz_bestfit = H_z_spline(a,b,c,d,zknots,zdat)
            Hz_bestfit[-1] = d[-1]
            #Samples within CL:
            Hz_CLsamples = []
            for i in range(Nsamples):
                a,b,c,d = spline_reconstruction(self.samples_lim_CL[i,:Nknots],h,R,Q)
                Hz_i = H_z_spline(a,b,c,d,zknots,zdat)
                Hz_i[-1] = d[-1]
                Hz_CLsamples.append(Hz_i)
        else:
            #bestfit case:
            params = {}
            params['H0'] = self.bestfit[0]
            params['Omega_m'] = self.bestfit[1]
            zdat = np.linspace(0,10,1000)
            if self.expansion == "flatwCDM":
                params['w0'] = self.bestfit[2]
            elif self.expansion == "flatw0waCDM":
                params['w0'] = self.bestfit[2]
                params['wa'] = self.bestfit[3]
            elif self.expansion == "LCDM":
                params['Omega_k'] = self.bestfit[2]
            elif self.expansion == "wCDM":
                params['w0'] = self.bestfit[2]
                params['Omega_k'] = self.bestfit[3]
            elif self.expansion == "w0waCDM":
                params['w0'] = self.bestfit[2]
                params['wa'] = self.bestfit[3]
                params['Omega_k'] = self.bestfit[4]
            Hz_bestfit = H_z(self,params,zdat)
            #Samples within CL
            Hz_CLsamples = []
            for i in range(Nsamples):
                params = {}
                params['H0'] = self.samples_lim_CL[i,0]
                params['Omega_m'] = self.samples_lim_CL[i,1]
                if self.expansion == "flatwCDM":
                    params['w0'] = self.samples_lim_CL[i,2]
                elif self.expansion == "flatw0waCDM":
                    params['w0'] = self.samples_lim_CL[i,2]
                    params['wa'] = self.samples_lim_CL[i,3]
                elif self.expansion == "LCDM":
                    params['Omega_k'] == self.samples_lim_CL[i,2]
                elif self.expansion == "wCDM":
                    params['w0'] = self.samples_lim_CL[i,2]
                    params['Omega_k'] == self.samples_lim_CL[i,3]
                elif self.expansion == "w0waCDM":
                    params['w0'] = self.samples_lim_CL[i,2]
                    params['wa'] = self.samples_lim_CL[i,3]
                    params['Omega_k'] == self.samples_lim_CL[i,4]
                Hz_CLsamples.append(H_z(self,params,zdat))
            
        self.Hz_bestfit = Hz_bestfit
        self.Hz_CLsamples = Hz_CLsamples
        
        return

    def print_param_indices(self):
        '''
        Prints the parameter indices for reference to analyze the MCMC
        '''
        #Add cosmo params
        params = []
        indices = []
        counter = 0
        if self.expansion == 'flexknot':
            Nknots = int(self.summary['knots info'].split('knots')[0][0])
            for i in range(Nknots-2):
                params.append('z_knot_'+str(i+1))
                indices.append(counter)
                counter += 1
            for i in range(Nknots):
                params.append('H(z_knot_'+str(i)+')')
                indices.append(counter)
                counter += 1
            if not self.summary['flat']:
                params.append('Omega_k')
                indices.append(counter)
                counter += 1
        elif self.expansion == 'spline':
            zknots = self.summary['knot position']
            Nknots = len(zknots)
            for i in range(Nknots):
                params.append('H(z_knot_'+str(i)+')')
                indices.append(counter)
                counter += 1
            if not self.summary['flat']:
                params.append('Omega_k')
                indices.append(counter)
                counter += 1
        else:
            params.append('H0')
            indices.append(counter)
            counter += 1
            params.append('Omega_m')
            indices.append(counter)
            counter += 1
            if self.expansion == "flatwCDM":
                params.append('w0')
                indices.append(counter)
                counter += 1
            elif self.expansion == "flatw0waCDM":
                params.append('w0')
                indices.append(counter)
                counter += 1
                params.append('wa')
                indices.append(counter)
                counter += 1
            elif self.expansion == "LCDM":
                params.append('Omega_k')
                indices.append(counter)
                counter += 1
            elif self.expansion == "wCDM":
                params.append('w0')
                indices.append(counter)
                counter += 1
                params.append('Omega_k')
                indices.append(counter)
                counter += 1
            elif self.expansion == "w0waCDM":
                params.append('w0')
                indices.append(counter)
                counter += 1
                params.append('wa')
                indices.append(counter)
                counter += 1
                params.append('Omega_k')
                indices.append(counter)
                counter += 1
        #Add nuisance parameters
        data = self.summary['data']
        if 'BAO' in data:
            params.append('rs')
            indices.append(counter)
            counter += 1
        if 'SN' in data:
            params.append('M')
            indices.append(counter)
            counter += 1

        for i in range(len(params)):
            print('Parameter Names and Indices:')
            print('----------------------------')
            print('{}  -> {}'.format(indices[i],params[i]))

        return 

    
    

    
