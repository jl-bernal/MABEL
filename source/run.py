'''
Base module for running the MCMCs
2021 - Jose Luis Bernal & Chih-Fan Chen
'''

import numpy as np
from numpy.random import normal
try:
    import zeus
    No_zeus = False
except:
    No_zeus = True
try:
    import emcee
    No_emcee = False
except:
    No_emcee = True
if No_zeus and No_emcee:
    raise ValueError('You need to have either emcee or zeus installed to run MABEL.')
    
import deepdish as dd

from source.utilities import get_default_params,check_params,check_lkls,check_zmax,spline_preparation
from source.read_data import read_and_process_data
from source.likelihoods import lnpost


class Run(object):
    '''
    An object controlling all relevant quantities needed to run the MCMCs
    and ease the use of MABEL to run MCMCs.
    
    It allows for three types of expansion models: standard (including 
    flatLCDM, flatwCDM, flatw0waCDM, LCDM, wCDM, w0waCDM), splines, and flexknots.
    It assumes one of these expansion models and run the MCMC using the likelihoods
    set by the input.
    
    INPUT PARAMETERS:
    ------------------
    
    sampler and MCMC options:
        -sampler            choose between zeus and emcee samplers. (default: zeus)
        
        -stepsize           stepsize for emcee (the "a" param in EnsembleSampler).
                            Only relevant if sampler == emcee
                            (default: 1.5)
                            
        -nsteps             Number of steps per walker (default: 5000)
        
        -save_inline        Bool. Save the chain as it progresses? So far only available
                            if sampler == emcee (Default: True)
        
    expansion:
        -expansion          Kind of expansion history assumed. Choose between splines,
                            flexknot, or one of the following: 'flatLCDM','flatwCDM',
                            'flatw0waCDM','LCDM',"wCDM","w0waCDM" (default: flatLCDM)
                            
        -flat               Bool. Whether the spatial section is flat or not. 
                            Only relevant if expansion == flexknot or spline 
                            (the flatness for the standard cosmo models is determined
                            by whether they are called 'flat*' or not (Default: True)
                            
        -zmax               Maximum z for flexknot expansion. Only relevant if 
                            expansion == flexknot; zmax = z_knots[-1] if 
                            expansion ==  spline, and zmax = 10 otherwise (default: 2.4)
        
        -Nknots             Number of knots for flexknot expansion. Only relvent
                            if expansion == flexknot. Nknots = len(z_knots) for 
                            expansion == spline, and irrelevant otherwise (default: 6)
                            
        -z_knots            Position of the knots for spline expansion.
                            Only relevant if expansion == spline.
                            (default: [0.,0.3,0.6,1.,1.5,2.4])
    
    likelihoods:            which likelihoods to include. Bool. Default: All false
        lkls                Dictionary including the bools for each likelihood.
         
                            Components: 
                                H0prior                     A prior on H0
                                BAO                         BAO from SDSS, WiggleZ and 6dFGRS
                                SN                          SNeIa from Pantheon
                                Clocks                      H(z) measurements from cosmic clocks
                                rdprior                     A prior on rd
                                          
        
    output and verbose:
        output_root         Name of the root for store outputs. 
                            (Default: output/default or output/[input_file] if a file
                            is used)
                            
        verbose_data        Print info about reading data (Default: False)
        
        verbose_run         Print info about the running and storing of the MCMC (Default: False)
                                
    '''
    def __init__(self,
                 sampler = 'zeus',stepsize = 1.5,nsteps = 5000,save_in_line=False,
                 expansion = 'flatLCDM',flat = True,
                 zmax = 2.4, Nknots = 6, z_knots = [0.,0.3,0.6,1.,1.5,2.4],
                 lkls = dict(H0prior = False, rdprior = False,BAO = False, SN = False, 
                             Clocks = False),
                 output_root = "output/default", verbose_data = False, verbose_run = False):
                 
        # Get list of input values to check type and units
        self._run_params = locals()
        self._run_params.pop('self')
        
        # Get list of input names and default values
        self._default_run_params = get_default_params(Run.__init__)
        # Check that input values have the correct type and units
        check_params(self._run_params,self._default_run_params)
        # Fill lkls no included with false
        for key in list(self._default_run_params['lkls'].keys()):
            if key not in self._run_params['lkls'].keys():
                self._run_params['lkls'][key] = False
        #check compatible likelihoods
        check_lkls(self._run_params['lkls'])
        
        # Set all given parameters
        for key in self._run_params:
            setattr(self,key,self._run_params[key])
                                
        #standard cosmo models
        self.std_cosmo_model_list=['flatLCDM','flatwCDM','flatw0waCDM','LCDM',"wCDM","w0waCDM"]
        if not (self.expansion in self.std_cosmo_model_list or self.expansion=="spline" or self.expansion=="flexknot"):
            print(self.std_cosmo_model_list)
            print(['spline', "flexknot"])
            raise ValueError('Please enter either one of types of the expansion history above')
            
        #check for save_in_line and zeus
        if self.save_in_line and self.sampler == 'zeus':
                print('Saving in line is only available for emcee for the moment.')
                self.save_in_line = False
                
        #transform z_knots to an array and set zmax
        if self.expansion == 'spline':
            self.z_knots = np.array(self.z_knots)
            self.zmax = self.z_knots[-1]
        elif self.expansion in self.std_cosmo_model_list:
            self.zmax = 10.
            
    def prep_data(self):
        '''
        Initialize all the needed data for the likelihoods of interest
        '''
        #print verbose
        if self.verbose_data:
            print("Initializing data!\nData included:")
            for key in self.lkls.keys():
                if self.lkls[key]:
                    print("\t",key)
            print("Type of expansion history assumed:", self.expansion)
            if self.expansion=="spline" or self.expansion=="flexknot":
                if self.flat:
                    print("Assuming a FLAT space-time")
                else:
                    print("Assuming a CURVED space-time")
        #Get data
        self.data, self.cov = read_and_process_data(self.lkls)
        #Check zmax is covered
        check_zmax(self)
        return
        
    def prep_initialcond(self):
        '''
        Determine number of parameters, initial positions and prior limits for the MCMC
        '''
        #set priors and initial positions
        if self.expansion == "flexknot":
            n_mat = self.Nknots
            n = 2*n_mat - 2 #positions of the first and last knots are fixed
            factor = 20
            if not self.flat:
                n += 1
        elif self.expansion == "spline":
            n_mat = len(self.z_knots)
            self.Nknots = n_mat
            n = n_mat
            factor = 20
            #compute spline preparation matrices
            self.h, self.R, self.Q = spline_preparation(self.z_knots)
            if not self.flat:
                n += 1
        else:
            factor = 10
            n = [2,3,4,3,4,5][self.std_cosmo_model_list.index(self.expansion)]
            n_cosmoparam = n

        #Add counting for nuisance parameters
        if self.lkls['BAO']:
            n += 1
        if self.lkls['SN']:
            n += 1
        
        nwalkers = n*factor
        
        if self.expansion == "flexknot":
            # positions of knots
            prior_min = [0.02 for i in range(n_mat - 2)]
            prior_max = [self.zmax - 0.02 for i in range(n_mat - 2)]
            pos = np.zeros((nwalkers, n_mat - 2))
            dummy_knot = np.linspace(0., self.zmax, self.Nknots)
            for i in range(n_mat - 2):
                pos[:, i] = np.ones(nwalkers) * dummy_knot[i + 1] + normal(0., 0.02, nwalkers)
            # initial position for H(z) around lcdm
            for i in range(n_mat):
                prior_min.append(0.)
                prior_max.append(1e4)
                pos = np.hstack((pos, (np.ones(nwalkers) * (70 * np.sqrt(0.31 * (1 + dummy_knot[i]) ** 3 + 1. - 0.31)) + 
                                        normal(0., 2.,nwalkers)).reshape((nwalkers, 1))))
            if not self.flat:
                prior_min.append(-500)
                prior_max.append(500)
                pos = np.hstack((pos, (np.zeros(nwalkers) + normal(0., .01, nwalkers)).reshape((nwalkers, 1))))

        elif self.expansion == "spline":
            prior_min = [0. for i in range(n_mat)]
            prior_max = [1e4 for i in range(n_mat)]
            pos = np.zeros((nwalkers, n_mat))
            # initial position for H(z) around lcdm
            for i in range(n_mat):
                pos[:, i] = np.ones(nwalkers) * (70 * np.sqrt(0.31 * (1 + self.z_knots[i]) ** 3 + 1. - 0.31)) + normal(0., 2.,nwalkers)
            if not self.flat:
                prior_min.append(-500)
                prior_max.append(500)
                pos = np.hstack((pos, (np.zeros(nwalkers) + normal(0., .01, nwalkers)).reshape((nwalkers, 1))))
        else:
            #Order (if param appearing) is: H0, Omega_m, w0, wa, Omega_k
            prior_min = [0.0, 0.]
            prior_max = [500, 1]
            pos = np.zeros((nwalkers, n_cosmoparam))
            pos[:, 0] = np.ones(nwalkers) * 70. + normal(0., 2., nwalkers)
            pos[:, 1] = np.ones(nwalkers) * 0.3 + normal(0., 0.02, nwalkers)
            if self.expansion == "flatLCDM":
                pass
                
            elif self.expansion == "flatwCDM":
                pos[:, 2] = np.ones(nwalkers) * -1 + normal(0., 0.02, nwalkers)
                prior_min.append(-2.5)
                prior_max.append(0.5)
            elif self.expansion == "flatw0waCDM":
                pos[:, 2] = np.ones(nwalkers) * -1 + normal(0., 0.02, nwalkers)
                prior_min.append(-2.5)
                prior_max.append(0.5)
                pos[:, 3] = np.ones(nwalkers) * 0 + normal(0., 0.02, nwalkers)
                prior_min.append(-10)
                prior_max.append(10)
            elif self.expansion == "LCDM":
                pos[:,2] = np.zeros(nwalkers) + normal(0., .01, nwalkers)
                prior_min.append(-500)
                prior_max.append(500)
            elif self.expansion == "wCDM":
                pos[:, 2] = np.ones(nwalkers) * -1 + normal(0., 0.02, nwalkers)
                prior_min.append(-2.5)
                prior_max.append(0.5)
                pos[:,3] = np.zeros(nwalkers) + normal(0., .01, nwalkers)
                prior_min.append(-500)
                prior_max.append(500)
            elif self.expansion == "w0waCDM":
                pos[:, 2] = np.ones(nwalkers) * -1 + normal(0., 0.02, nwalkers)
                prior_min.append(-2.5)
                prior_max.append(0.5)
                pos[:, 3] = np.ones(nwalkers) * 0 + normal(0., 0.02, nwalkers)
                prior_min.append(-10)
                prior_max.append(10)
                pos[:,4] = np.zeros(nwalkers) + normal(0., .01, nwalkers)
                prior_min.append(-500)
                prior_max.append(500)
                
        if self.lkls['BAO']:
            prior_min.append(0)
            prior_max.append(500)
            pos = np.hstack((pos, (np.ones(nwalkers) * 150. + normal(0., 5., nwalkers)).reshape((nwalkers, 1))))
        if self.lkls['SN']:
            prior_min.append(-1e3)
            prior_max.append(1e3)
            pos = np.hstack((pos, (np.ones(nwalkers) * -19. + normal(0., 1., nwalkers)).reshape((nwalkers, 1))))
        
        self.npars = n 
        self.nwalkers = nwalkers  
        self.prior_min = prior_min
        self.prior_max = prior_max
        self.pos = pos  
        return
        
    
            
    def run_mcmc(self):
        '''
        Call the sampler of choice and run the MCMC
        '''
        if self.sampler == 'zeus':
            zeus_sampler = zeus.EnsembleSampler(self.nwalkers, self.npars, lnpost,
                args=[self])
                
            if self.verbose_run:
                print('{} parameters, {} walkers'.format(self.npars,self.nwalkers))
                print('Total steps = ', self.nwalkers*self.nsteps)
                print("Running MCMC...")
                progress = True
            else:
                progress = False
                
            zeus_sampler.run_mcmc(self.pos, self.nsteps,progress=progress) # Run sampling
            self.MCMC = zeus_sampler
            
            
        elif self.sampler == 'emcee':
            backend = emcee.backends.HDFBackend(self.output_root + '.h5')
            backend.reset(self.nwalkers, self.npars)

            emcee_sampler = emcee.EnsembleSampler(self.nwalkers, self.npars, lnpost, a=self.stepsize,
                                            args=[self],
                                            backend=backend)
            if self.verbose_run:
                print('{} parameters, {} walkers'.format(self.npars,self.nwalkers))
                print('Total steps = ', self.nwalkers*self.nsteps)
                print("Running MCMC...")
                progress = True
            else:
                progress = False
                
            if self.save_in_line:
                index = 0
                autocorr = np.empty(self.nsteps)
                old_tau = np.inf
                # Now, sample for up to max_steps
                for sample in emcee_sampler.sample(self.pos, iterations=self.nsteps, progress=progress):
                    # Only check convergence every 200 steps
                    if emcee_sampler.iteration % 200:
                        continue
                    # Compute the autocorrelation time so far
                    # Using tol=0 means that we'll always get an estimate even
                    # if it isn't trustworthy
                    tau = emcee_sampler.get_autocorr_time(tol=0)
                    autocorr[index] = np.mean(tau)
                    index += 1
                    # Check convergence
                    converged = np.all(tau * 100 < emcee_sampler.iteration)
                    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                    if self.verbose_run:
                        print('SO FAR:  Acceptance fraction = {:.2f} (should be between 0.2 and 0.5)'.format(
                            np.sum(emcee_sampler.acceptance_fraction) / self.nwalkers))
                    if converged:
                        break
                    old_tau = tau
            else:
                emcee_sampler.run_mcmc(self.pos, self.nsteps, progress=progress)
            
            self.MCMC = emcee_sampler
        return
            
            
    def save_mcmc(self):
        '''
        Save the products of the MCMC with deepdish to use later
        '''
        SUMMARY = {}
        SUMMARY['expansion model'] = self.expansion
        datavec = ''
        for key in list(self.lkls):
            if self.lkls[key]: 
                datavec += key+'   '
        if self.expansion == 'flexknot':
            SUMMARY['flat'] = self.flat
            SUMMARY['knots info'] = '{} knots, zmax = {}'.format(self.Nknots,self.zmax)
        elif self.expansion == 'spline':
            SUMMARY['flat'] = self.flat
            SUMMARY['knot position'] = self.z_knots        
        elif 'flat' in self.expansion:
            SUMMARY['flat'] = True
        else:
            SUMMARY['flat'] = False
        SUMMARY['data'] = 'Datasets used: {}'.format(datavec)
        if self.sampler == 'zeus':
            try:
                tau = self.MCMC.act
            except:
                tau = 'There was an issue computing the autocorrelation time; please check your zeus and/or scipy versions!'#burnin = int(2 * np.mean(tau))

            SUMMARY['chain stats'] ='''
            MCMC stats:
            -------------------------
            Number of Generations: {}
            Number of Parameters: {}
            Number of Walkers: {}
            Number of Tuning Generations: {}
            Scale Factor: {}
            Mean Integrated Autocorrelation Time: {}
            Effective Sample Size: {}
            Number of Log Probability Evaluations: {}
            Effective Samples per Log Probability Evaluation: {}
            '''.format(self.MCMC.samples.length,self.MCMC.ndim,self.MCMC.nwalkers,len(self.MCMC.mus),
            round(self.MCMC.mu,6),round(np.mean(self.MCMC.act),2),round(self.MCMC.ess,2),
            self.MCMC.ncall,round(self.MCMC.efficiency,6))#,burnin)
            SUMMARY['Correlation time'] = tau
            #Save the chain
            samples = self.MCMC.get_chain(discard=0, flat=False, thin=1)
            log_prob_samples = self.MCMC.get_log_prob(discard=0, flat=False, thin=1)

            DICT = {'samples':samples,'log_prob_samples':log_prob_samples,'Summary':SUMMARY}
            dd.io.save(self.output_root+'.h5', DICT)
        elif self.sampler == 'emcee':
            SUMMARY['chain stats'] ='''
            MCMC stats:
            -------------------------
            Acceptance_fraction = {}
            Mean autocorrelation time = {}
            '''.format(np.mean(self.MCMC.acceptance_fraction),round(np.mean(self.MCMC.get_autocorr_time()),2))
            SUMMARY['Correlation time'] = self.MCMC.get_autocorr_time()
            #Save the chain
            samples = self.MCMC.get_chain(discard=0, flat=False, thin=1)
            log_prob_samples = self.MCMC.get_log_prob(discard=0, flat=False, thin=1)

            DICT = {'samples':samples,'log_prob_samples':log_prob_samples,'Summary':SUMMARY}
            dd.io.save(self.output_root+'.h5', DICT)
        return

        
    def run_MABEL(self,save=True):
        '''
        Instance to run the whole process of MABEL with one line. It wrappes:
        prep_data(), prep_initialcond(), run_mcmc() and, if save==True, save_mcmc()
        '''
        self.prep_data()
        self.prep_initialcond()
        self.run_mcmc()
        if save:
            self.save_mcmc()
        return
            
            
            
            
            
            
            
            
            
            
            

