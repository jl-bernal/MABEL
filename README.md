# MABEL: Model-Agnostic Background Expansion Likelihoods

MABEL is a python code to constrain the Hubble expansion rate at low redshift with cosmological observations, using parametrizations of H(z) that are agnostic of any specific cosmological model. MABEL allows to choose between [zeus](https://github.com/minaskar/zeus) or [emcee](https://emcee.readthedocs.io/en/stable/) as samplers. MABEL also includes routines to analyze the MCMC and reconstruct H(z) from each of the samples in the MCMC.

Conveniently, MABEL includes many useful and common models and likelihoods, but also has a structure that makes very easy to implement new models or likelihoods. 

## Models

The main goal of MABEL is to provide a unified framework for parameter inference and constraining the expansion history at low redshift using agnostic approaches. In all cases, it is possible to impose flatness or free that assumption. Currently the parametrizations included are:

- spline: Interpolation of H(z) using natural cubic splines. The free parameters are H(z_knot), where z_knot are fixed points (given by the input) to interpolate H(z).

- flexknot: Similar to "spline", but in this case the position of z_knot also varies. Hence, the free parameters are H(z_knot) and z_knot (but the first and last entries in z_knot, which are fixed to 0 and z_max (given in the input)). 

In addition, the standard flat LCDM and common extensions affecting the expansion history at low redshift are included:

- flatLCDM, flatwCDM, flatw0waCDM, LCDM, wCDM, w0waCDM.

#### Including your own model:

In order to include your own model for the expansion history, you can make it in an analogous way as e.g., LCDM is implemented. Follow the steps below:

- In `source/cosmo.py`: Add the computation of H(z) in `H_z()`.

- In `source/run.py`: Include the initialization of parameters and the hard priors in `prep_initialcond()`. Also, add the model name to the `std_cosmo_model_list`.

- In `source/utilities.py`: Include the new parameters in the params dictionary in `assign_params()`.

- In `source/analyze.py`: Add it to the `expansion_list` and also list the parameters of the model in `print_param_indices()`.

## Likelihoods

Currently, MABEL includes the following likelihoods:

- H0prior: A prior on Hubble constant H_0 e.g., from SHOES or CCHP.

- rdprior: A prior on the sound horizon rd at radiation drag, from e.g., Planck.

- BAO: BAO from 6dF, SDSS DR7 MGS, WiggleZ, BOSS and eBOSS.

- SN: SNeIa from Pantheon.

- Clocks: H(z) measurements from cosmic clocks.

We will include more likelihoods in the future.

#### Including your own likelihood:

In order to include your likelihood to constrain H(z), you can make it in an analogous way as some of the already implemented likelihoods. Follow the steps below:

- In `source/likelihoods.py`: Include the likelihood as a function returning the chi2, and also include a call to that function in `ln_lkls()` in case the likelihood is included in the input (follow the example of already implemented likelihoods).

- Create a data folder in `data/` with the name of the likelihood and include there all the data files needed (note that the same folder can be used by more than one likelihood, as called by `read_and_process_data()`, see below).

- In `source/read_data.py`: Indicate how to read the data from the data files in the `data/` folder in `read_and_process_data()`.

- In `source/utilities.py`: Add eventual incompatibilities between likelihoods in `check_lkls()`, and a check to be sure that the highest redshift of the likelihood is covered by the model in `check_zmax()`.

- In case the likelihood has nuisance parameters:
    - In `source/run.py`: Include the initialization of parameters and the hard priors in `prep_initialcond()`.
    - In `source/utilities.py`: Include the new parameters in the params dictionary in `assign_params()`.
    - In `source/analyze.py`: List the parameters of the model in `print_param_indices()`.
    
## Installation and use

Clone this repository and include the path to the main folder in your PYTHONPATH. To check that it has been link, try:

```
from mabel import mabel
```

You can find examples of how to use MABEL, both for running and analyzing chains, in the `notebooks/` folder.

#### Prerequisites:

Besides standard packages as numpy, scipy, os, pandas and matplotlib, MABEL uses:

- PyYAML (if you want to use input files; otherwise can input the parameters with a python dictionary).
- At least one of the following: [zeus](https://github.com/minaskar/zeus) or [emcee](https://emcee.readthedocs.io/en/stable/).
- deepdish.
- numexpr.

## Usage

You are free to use MABEL in your research, but please refer to this GitHub repository and cite:

- [The trouble beyond H0 and the new cosmic triangles. arXiv:2102.05066](https://arxiv.org/abs/2102.05066)
- The references related to either [zeus](https://github.com/minaskar/zeus) or [emcee](https://emcee.readthedocs.io/en/stable/), depending the sampler that you used.

## Main contributors

* **José Luis Bernal**
* **Geoff Chih-Fan Chen**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details






