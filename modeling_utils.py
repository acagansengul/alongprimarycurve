import func_utils as fu
import numpy as np
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit

#defining numerical angular deflection class##################    
class CustomLensingClass(object):
    def __init__(self,alpha_map):
        self.alpha_map = alpha_map  
        return None

    def __call__(self, x, y,**kwargs):
        alpha_map = self.alpha_map
        return alpha_map(x,y)
#############################################################


def linearfit_eigencurve_basis_deflection_image(source_params,lens_params,data_class,psf_class,source_model_class,likemask):
    curve_params = lens_params['curve_params'] 
    lambda_1poly = lens_params['lambda_1poly']
    lambda_2poly = lens_params['lambda_2poly']
    Fpoly = lens_params['Fpoly']
    H0 = lens_params['H0']

    lens_func = fu.make_lens_func(curve_params,lambda_1poly,lambda_2poly,Fpoly,H0)

    def alpha_func(x,y):
        alpha_x = np.zeros(len(x))
        alpha_y = np.zeros(len(x))
        for i in range(len(x)):
            alpha_x[i],alpha_y[i] = np.array([x[i],y[i]]) - np.array(lens_func(x[i],y[i]))
        return np.array([alpha_x,alpha_y])
    
    ################################################################################
    custom_class = CustomLensingClass(alpha_map = alpha_func)
    lens_model_class = LensModel(lens_model_list=['TABULATED_DEFLECTIONS'],numerical_alpha_class=custom_class)
    
    empty_light_model = LightModel([])
    empty_light_param = []

    kwargs_numerics = {'supersampling_factor': 1}
    
    imageModel = ImageLinearFit(data_class=data_class, psf_class=psf_class, kwargs_numerics=kwargs_numerics, 
                                lens_model_class=lens_model_class, source_model_class=source_model_class,
                                lens_light_model_class = empty_light_model,
                                likelihood_mask=likemask)
    
    wls_model, model_error, cov_param, coeffsq = imageModel.image_linear_solve([{}], source_params, 
                                                                   kwargs_lens_light=empty_light_param, kwargs_ps=None, inv_bool=False)
    
    return wls_model, coeffsq,lens_func
    
    
    
def create_eigencurve_basis_deflection_image(source_params,lens_params,data_class,psf_class,source_model_class):
    curve_params = lens_params['curve_params'] 
    lambda_1poly = lens_params['lambda_1poly']
    lambda_2poly = lens_params['lambda_2poly']
    Fpoly = lens_params['Fpoly']
    H0 = lens_params['H0']

    lens_func = fu.make_lens_func(curve_params,lambda_1poly,lambda_2poly,Fpoly,H0)

    def alpha_func(x,y):
        alpha_x = np.zeros(len(x))
        alpha_y = np.zeros(len(x))
        for i in range(len(x)):
            alpha_x[i],alpha_y[i] = np.array([x[i],y[i]]) - np.array(lens_func(x[i],y[i]))
        return np.array([alpha_x,alpha_y])
    
    ################################################################################
    custom_class = CustomLensingClass(alpha_map = alpha_func)
    lens_model_class = LensModel(lens_model_list=['TABULATED_DEFLECTIONS'],numerical_alpha_class=custom_class)
    
    empty_light_model = LightModel([])
    empty_light_param = []

    kwargs_numerics = {'supersampling_factor': 1}
    
    imageModel = ImageModel(data_class=data_class, psf_class=psf_class, kwargs_numerics=kwargs_numerics, 
                                lens_model_class=lens_model_class, source_model_class=source_model_class,
                                lens_light_model_class = empty_light_model)
    
    wls_model = imageModel.image([{}], source_params, kwargs_lens_light=empty_light_param, kwargs_ps=None)
    
    return wls_model
    
    

def visualize_shapelet_source(amps,n_max,beta,xvals,yvals):
    ext = np.max([np.max(yvals)-np.min(yvals),np.max(xvals)-np.min(xvals)])
    numPix2 = 500
    deltaPix = ext/numPix2
    
    source_params =  [{'amp':amps,'n_max': n_max, 'beta': beta, 'center_x': 0., 'center_y': 0.}]
    exp_time = 1.
    sigma_bkg = 1.
    psf_type = 'NONE'  # 'GAUSSIAN', 'PIXEL', 'NONE'
    kwargs_psf = {'psf_type': psf_type,'fwhm':2*deltaPix}
    psf_class2 = PSF(**kwargs_psf)
    kwargs_numerics = {'supersampling_factor': 1}
    kwargs_data2 = sim_util.data_configure_simple(numPix2, deltaPix, exp_time, sigma_bkg)
    data_class2 = ImageData(**kwargs_data2)
    #############################################################
    empty_light_model = LightModel([])
    empty_light_param = []
    
    source_model_type = ['SHAPELETS_POLAR']
    source_model_class = LightModel(light_model_list=source_model_type)

    lens_model_list_new = ['SHEAR']
    lens_kwargs = [{'gamma1': 0., 'gamma2': 0.}]  
    lens_model_class = LensModel(lens_model_list=lens_model_list_new)

        # initialize the Image model class by combining the modules we created above #
    imageModel = ImageModel(data_class=data_class2, psf_class=psf_class2, lens_model_class=lens_model_class,
                            source_model_class=source_model_class,
                            lens_light_model_class=empty_light_model,
                            point_source_class=None, # in this example, we do not simulate point source.
                            kwargs_numerics=kwargs_numerics)
    # simulate image with the parameters we have defined above #
    source_image = imageModel.image(kwargs_lens=lens_kwargs, kwargs_source=source_params,
                             kwargs_lens_light=empty_light_param, kwargs_ps=None)
    
    return source_image