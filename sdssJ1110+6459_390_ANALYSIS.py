#################### Loading necessary packages
import numpy as np
from numpy.polynomial import Polynomial

import dynesty
import pyswarms as ps
from multiprocessing import Pool

import func_utils as fu
import modeling_utils as mu
import modeling_utils_triangle as mu_trig


import dynesty
import pyswarms as ps
from multiprocessing import Pool


flat390 = np.load('analysis_data/flat390.npy')
coords390 = np.load('analysis_data/coords390.npy')
likemask390 = np.load('analysis_data/likemask390.npy')
scord390 = np.load('analysis_data/scord390.npy')
triplets390 = np.load('analysis_data/triplets390.npy')
psf390 = np.load('analysis_data/psf390.npy')


nparal = 48

psf390 = psf390/np.sum(psf390)
deltaPix = 0.03962000086903572
sigma = 0.002823125913
psf_matrix = mu_trig.make_psf_matrix(psf390,flat390,coords390,deltaPix)


npar = 7

curvemax = [+2.0, +0.5, 0.3]
curvemin = [-2.0, -0.5,-0.3]

lensmax = [+4.,  0., 1., 50.]
lensmin = [-4., -1., 0., -50.]

minparam = curvemax + lensmax
maxparam = curvemin + lensmin

ct1 = -0.9
ct2 = -2.5
theta0 = np.pi/2.+0.03
a_array = np.array([-0.055,0.0,0.0])

t_cr0 = 7.7


def lnlike_simple_FAST(params):
    ########################### CURVE PARAMS
    theta0val = theta0 + params[1]
    d1 = ct1 + params[0]*np.cos(theta0val)
    d2 = ct2 + params[0]*np.sin(theta0val)
    a_array_fit = a_array + np.array([params[2],0.,0.])
    curve_params = {'d1':d1,
                    'd2':d2,
                    'theta0':theta0val,
                    'a_array':a_array_fit}
    ########################### L1 PARAMS
    dt_cr = params[3]
    r_0 = params[4]
    r_cr = params[5]
    q4 = params[6]/(t_cr0**4)
    q_extra = [0.]
    lambda_1poly = fu.makeQm(t_cr0 + dt_cr,r_0,r_cr,q4,q_extra)
    ########################### L2 PARAMS
    l2_0 = 1.
    lambda_2poly = Polynomial(np.array([l2_0]))
    ########################### F PARAMS
    F_0 = 0.
    Fpoly = Polynomial(np.array([F_0]))
    ########################### H0 PARAMS
    H0 = 0.
    ########################### FULL LENS PARAMS
    lens_params = {'curve_params':curve_params,
                   'lambda_1poly':lambda_1poly,
                   'lambda_2poly':lambda_2poly,
                   'Fpoly':Fpoly,
                   'H0':H0}

    amps, models, lensfunc0,delaunay = mu_trig.linearfit_eigencurve_basis_deflection_image_FAST(lens_params,flat390,scord390,coords390,psf_matrix,deltaPix)

    res = flat390 - models
    sqrd = (res/sigma)**2.
    
    return -0.5*(np.sum(sqrd))


# Define our uniform prior.
def ptform(u):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""
    return minparam*(1.-u) + maxparam*u 
    
    
pool = Pool(nparal)
    
# initialize our sampler
sampler = dynesty.DynamicNestedSampler(lnlike_simple_FAST, ptform, npar, pool=pool, queue_size=nparal)
# run the sampler with checkpointing
#sampler = dynesty.NestedSampler.restore('tripoint_optim4.save')
sampler.run_nested(checkpoint_file='sdss_J1110_f390w_NO9POINT_CURVE3_LENS4.save')