'''
Primary module to call MABEL
2021 - Jose Luis Bernal & Chih-Fan Chen
'''

from source.run import Run
from source.analyze import Analyze

try:
    import yaml
except:
    print('Warning, you will only be able to input parameters with a dictionary')


def mabel(input_pars=None):
    '''
    Base function to initiate mabel objects. 
    
    -input_pars:    String containing the path to a input file (*.yaml) or a
                    dictionary containing the input parameters. 
    '''

    #check input parameters given, and read them
    if type(input_pars) == str:
        with open(input_pars) as f:
            pars = yaml.load(f, Loader=yaml.FullLoader)
        if 'output_root' not in pars:
            pars['output_root'] = 'output/'+input_pars.split('/')[-1].split('.')[0]
    elif type(input_pars) == dict:
        pars = input_pars
    else:
        raise ValueError('Please input a dictionary or a path to a *.yaml file with the input parameters.')
    
    #Return the run object
    return Run(**pars)


def mabel_analyze(path,expansion, CL=0.6829):
    '''
    Base function to initiate mabel analyze objects.
    
    Input paameters:

        -path:      path to the file containing the MCMC
        
        -expansion: Type of expansion assumed in the MCMC run 
                    (choose between: 'flatLCDM','flatwCDM','flatw0waCDM','LCDM',"wCDM","w0waCDM")
                    
        -CL:        Confidence level (over 1) within to draw the samples
                    (default: 0.6829, 1sigma)
    '''
    return Analyze(path,expansion,CL)
