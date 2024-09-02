import numpy as np
from numpy.polynomial import Polynomial


################# Lambda_1 Functions
def makeQ3(t_cr,r_0,r_cr):
    ## returns the coefs unique degree 3 polynomial
    coefs = np.zeros(4)
    coefs[1] = r_0
    coefs[2] = r_0*(-1)*(1./t_cr)*(2. + (r_cr/r_0))
    coefs[3] = r_0*(1./(t_cr**2.))*(1 + (r_cr/r_0))

    Q3 = Polynomial(coefs)
    return Q3
    
def makeQ4(t_cr,q4):
    ## returns the 1dof degree 4 polynomial
    coefs = np.zeros(5)
    coefs[2] = q4
    coefs[3] = q4*(-2./t_cr)
    coefs[4] = q4*(1./t_cr**2)

    Q4 = Polynomial(coefs)
    return Q4

def makeQm(t_cr,r_0,r_cr, q4 = None,q_extra = []):
    if q4 == None:
        return makeQ3(t_cr,r_0,r_cr)
    elif len(q_extra) == 0:
        Q3 = makeQ3(t_cr,r_0,r_cr)
        Q4 = makeQ4(t_cr,q4)
        return Q3 + Q4
    else:
        Q3 = makeQ3(t_cr,r_0,r_cr)
        Q4 = makeQ4(t_cr,q4)
        const = Polynomial(np.array([1.]))
        id = Polynomial(np.array([0.,1.]))

        Q5 = Polynomial(q_extra)
        Qa = Q5 * id
        Qb = Qa + const
        Qc = Qb * Q4
        
        return Q3 + Qc
        
        
################### Curve Functions
def make_zeta(d1,d2,theta0,a_array):
    unit = Polynomial(np.array([1.]))
    iden = Polynomial(np.array([0.,1.]))
    Pbase = Polynomial(a_array)
    theta_n = Pbase * iden

    dP =  theta_n
    dP += -(1./6.)*(theta_n * theta_n * theta_n)
    dP += (1./120.)*(theta_n * theta_n * theta_n * theta_n * theta_n)
#    dP += -(1./5040.)*(theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dP += (1./362880)*(theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)

    P = dP.integ()

    dQ = unit
    dQ += -(1./2.)*(theta_n * theta_n)
    dQ += (1./24.)*(theta_n * theta_n * theta_n * theta_n)
#    dQ += -(1./720.)*(theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dQ += (1./40320.)*(theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
    
    Q = dQ.integ()

    def zeta(t,s):
        dvec = np.array([d1,d2])
        e1 = np.array([np.cos(theta0),np.sin(theta0)])
        e2 = np.array([-np.sin(theta0),np.cos(theta0)])
        return np.array([dvec[0] + (Q(t)-np.sin(theta_n(t))*s)*e1[0] + (P(t)+ np.cos(theta_n(t))*s)*e2[0],dvec[1] + (Q(t)-np.sin(theta_n(t))*s)*e1[1] + (P(t)+ np.cos(theta_n(t))*s)*e2[1]])
    return zeta
    
# implement quick inversion here.
def make_inverse_zeta(d1,d2,theta0,a_array):
    unit = Polynomial(np.array([1.]))
    iden = Polynomial(np.array([0.,1.]))
    Pbase = Polynomial(a_array)
    theta_n = Pbase * iden

    dP =  theta_n
    dP += -(1./6.)*(theta_n * theta_n * theta_n)
    dP += (1./120.)*(theta_n * theta_n * theta_n * theta_n * theta_n)
#    dP += -(1./5040.)*(theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dP += (1./362880)*(theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)

    P = dP.integ()

    dQ = unit
    dQ += -(1./2.)*(theta_n * theta_n)
    dQ += (1./24.)*(theta_n * theta_n * theta_n * theta_n)
#    dQ += -(1./720.)*(theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dQ += (1./40320.)*(theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
    
    Q = dQ.integ()
    
    
    def inverse_zeta(x,y):
        hatx = (x-d1)*np.cos(theta0) + (y-d2)*np.sin(theta0)
        haty = (y-d2)*np.cos(theta0) - (x-d1)*np.sin(theta0)

        titer = hatx
        Dtheta = theta_n.deriv()
        
 #       Df = 2.*(Q - hatx)*Q.deriv() + 2.*(P - haty)*P.deriv()
        
        for i in range(8):
            Qtiter = Q(titer)
            Ptiter = P(titer)
            theta_ntiter = theta_n(titer)
            titer += -(np.cos(theta_ntiter)*(Qtiter-hatx) + np.sin(theta_ntiter)*(Ptiter-haty))/(1. - Dtheta(titer)*(np.sin(theta_ntiter)*(Qtiter-hatx) - np.cos(theta_ntiter)*(Ptiter-haty)))
            #titer += -Df(titer)/Df.deriv()(titer)

        t = titer
        s = (haty - P(t))/np.cos(theta_n(t))
        return np.array([t,s])
    
    return inverse_zeta
    
####################### Lens Functions

def make_yfunc(curve_params,lambda_1,lambda_2,Fpoly, H0):
    ## curve_params is a dictionary of curve values
    d1 = curve_params['d1']
    d2 = curve_params['d2']
    theta0 = curve_params['theta0']
    a_array = curve_params['a_array']

    iden = Polynomial(np.array([0.,1.]))
    Pbase = Polynomial(a_array)
    theta_n = Pbase * iden

    dSlambda_1cos = lambda_1
    dSlambda_1cos += -(1./2.)*(lambda_1 * theta_n * theta_n)
    dSlambda_1cos += (1./24.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1cos += -(1./720.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1cos += (1./40320.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
    Slambda_1cos = dSlambda_1cos.integ() 
 
    dSlambda_1sin = (lambda_1 * theta_n)
    dSlambda_1sin += -(1./6.)*(lambda_1 * theta_n * theta_n * theta_n)
    dSlambda_1sin += (1./120.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1sin += -(1./5040.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1sin += (1./362880)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
    Slambda_1sin = dSlambda_1sin.integ()
     
    Ipoly = Fpoly.deriv() + 3.*(lambda_2.deriv() * theta_n.deriv())
    Hpoly = -4.*(Ipoly * theta_n.deriv()).integ() + H0
    
    def yfunc(t,s):
        e1 = np.array([np.cos(theta0),np.sin(theta0)])
        e2 = np.array([-np.sin(theta0),np.cos(theta0)])
        
        term1 = Slambda_1cos(t) + s*0.
        term2 = Slambda_1sin(t) + s*0.
        s0term = np.array([term1*e1[0] + term2*e2[0], term1*e1[1] + term2*e2[1]])

        sin = np.sin(theta_n(t))
        cos = np.cos(theta_n(t))
        
        fact = lambda_2(t)*s
        term3 = -sin*fact
        term4 = cos*fact
        s1term = np.array([term3*e1[0] + term4*e2[0], term3*e1[1] + term4*e2[1]])

        otherfact = Fpoly(t)*(0.5*(s**2.))
        term5 = -sin*otherfact
        term6 = cos*otherfact
        s2term = np.array([term5*e1[0] + term6*e2[0], term5*e1[1] + term6*e2[1]])

        otherfact2 = lambda_2.deriv()(t)*(0.5*(s**2.))
        term7 = cos*otherfact2
        term8 = sin*otherfact2
        s3term = np.array([term7*e1[0] + term8*e2[0], term7*e1[1] + term8*e2[1]])
        
        lastfact = Hpoly(t)*((1./6.)*(s**3.))
        term9 = -sin*lastfact
        term10 = cos*lastfact
        s4term = np.array([term9*e1[0] + term10*e2[0], term9*e1[1] + term10*e2[1]])
        
        lastfact2 = Ipoly(t)*((1./6.)*(s**3.))
        term11 = cos*lastfact2
        term12 = sin*lastfact2
        s5term = np.array([term11*e1[0] + term12*e2[0], term11*e1[1] + term12*e2[1]])
        
        return s0term + s1term + s2term + s3term + s4term + s5term

    return yfunc
    
def make_lens_func(curve_params,lambda_1,lambda_2,Fpoly, H0):
    d1 = curve_params['d1']
    d2 = curve_params['d2']
    theta0 = curve_params['theta0']
    a_array = curve_params['a_array']

    
    yfunc = make_yfunc(curve_params,lambda_1,lambda_2,Fpoly, H0)
    invzt = make_inverse_zeta(d1,d2,theta0,a_array)

    def lensfunc(x,y):
        return yfunc(invzt(x,y)[0],invzt(x,y)[1])

    return lensfunc
    
    
def make_yfunc_fast(curve_params,lambda_1,lambda_2):
    ## Simple fast yfunc with constant lambda2
    ## curve_params is a dictionary of curve values
    d1 = curve_params['d1']
    d2 = curve_params['d2']
    theta0 = curve_params['theta0']
    a_array = curve_params['a_array']

    iden = Polynomial(np.array([0.,1.]))
    Pbase = Polynomial(a_array)
    theta_n = Pbase * iden

    dSlambda_1cos = lambda_1
    dSlambda_1cos += -(1./2.)*(lambda_1 * theta_n * theta_n)
    dSlambda_1cos += (1./24.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1cos += -(1./720.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1cos += (1./40320.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
    Slambda_1cos = dSlambda_1cos.integ() 
 
    dSlambda_1sin = (lambda_1 * theta_n)
    dSlambda_1sin += -(1./6.)*(lambda_1 * theta_n * theta_n * theta_n)
    dSlambda_1sin += (1./120.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1sin += -(1./5040.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1sin += (1./362880)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
    Slambda_1sin = dSlambda_1sin.integ()

    
    def yfunc(t,s):
        e1 = np.array([np.cos(theta0),np.sin(theta0)])
        e2 = np.array([-np.sin(theta0),np.cos(theta0)])
        
        term1 = Slambda_1cos(t) + s*0.
        term2 = Slambda_1sin(t) + s*0.
        s0term = np.array([term1*e1[0] + term2*e2[0], term1*e1[1] + term2*e2[1]])

        sin = np.sin(theta_n(t))
        cos = np.cos(theta_n(t))
        
        fact = lambda_2*s
        term3 = -sin*fact
        term4 = cos*fact
        s1term = np.array([term3*e1[0] + term4*e2[0], term3*e1[1] + term4*e2[1]])
        
        return s0term + s1term

    return yfunc
    
def make_lens_func_fast(curve_params,lambda_1,lambda_2):
    d1 = curve_params['d1']
    d2 = curve_params['d2']
    theta0 = curve_params['theta0']
    a_array = curve_params['a_array']

    
    yfunc = make_yfunc_fast(curve_params,lambda_1,lambda_2)
    invzt = make_inverse_zeta(d1,d2,theta0,a_array)

    def lensfunc(x,y):
        return yfunc(invzt(x,y)[0],invzt(x,y)[1])

    return lensfunc
    
    
################## FREE LENS FUNC (NO 0 CURL REQUIREMENT)

def make_yfunc_free(curve_params,lambda_1,lambda_2,Fpoly, Gpoly, Hpoly, Ipoly):
    ## curve_params is a dictionary of curve values
    d1 = curve_params['d1']
    d2 = curve_params['d2']
    theta0 = curve_params['theta0']
    a_array = curve_params['a_array']

    iden = Polynomial(np.array([0.,1.]))
    Pbase = Polynomial(a_array)
    theta_n = Pbase * iden

    dSlambda_1cos = lambda_1
    dSlambda_1cos += -(1./2.)*(lambda_1 * theta_n * theta_n)
    dSlambda_1cos += (1./24.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1cos += -(1./720.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1cos += (1./40320.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
    Slambda_1cos = dSlambda_1cos.integ() 
 
    dSlambda_1sin = (lambda_1 * theta_n)
    dSlambda_1sin += -(1./6.)*(lambda_1 * theta_n * theta_n * theta_n)
    dSlambda_1sin += (1./120.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1sin += -(1./5040.)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
#    dSlambda_1sin += (1./362880)*(lambda_1 * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n * theta_n)
    Slambda_1sin = dSlambda_1sin.integ()
     
    
    def yfunc(t,s):
        e1 = np.array([np.cos(theta0),np.sin(theta0)])
        e2 = np.array([-np.sin(theta0),np.cos(theta0)])
        
        term1 = Slambda_1cos(t) + s*0.
        term2 = Slambda_1sin(t) + s*0.
        s0term = np.array([term1*e1[0] + term2*e2[0], term1*e1[1] + term2*e2[1]])

        sin = np.sin(theta_n(t))
        cos = np.cos(theta_n(t))
        
        fact = lambda_2(t)*s
        term3 = -sin*fact
        term4 = cos*fact
        s1term = np.array([term3*e1[0] + term4*e2[0], term3*e1[1] + term4*e2[1]])

        otherfact = Fpoly(t)*(0.5*(s**2.))
        term5 = -sin*otherfact
        term6 = cos*otherfact
        s2term = np.array([term5*e1[0] + term6*e2[0], term5*e1[1] + term6*e2[1]])

        otherfact2 = Gpoly(t)*(0.5*(s**2.))
        term7 = cos*otherfact2
        term8 = sin*otherfact2
        s3term = np.array([term7*e1[0] + term8*e2[0], term7*e1[1] + term8*e2[1]])
        
        lastfact = Hpoly(t)*((1./6.)*(s**3.))
        term9 = -sin*lastfact
        term10 = cos*lastfact
        s4term = np.array([term9*e1[0] + term10*e2[0], term9*e1[1] + term10*e2[1]])
        
        lastfact2 = Ipoly(t)*((1./6.)*(s**3.))
        term11 = cos*lastfact2
        term12 = sin*lastfact2
        s5term = np.array([term11*e1[0] + term12*e2[0], term11*e1[1] + term12*e2[1]])
        
        return s0term + s1term + s2term + s3term + s4term + s5term

    return yfunc
    
def make_lens_func_free(curve_params,lambda_1,lambda_2, Fpoly, Gpoly, Hpoly, Ipoly):
    d1 = curve_params['d1']
    d2 = curve_params['d2']
    theta0 = curve_params['theta0']
    a_array = curve_params['a_array']

    
    yfunc = make_yfunc_free(curve_params,lambda_1,lambda_2,Fpoly, Gpoly, Hpoly, Ipoly)
    invzt = make_inverse_zeta(d1,d2,theta0,a_array)

    def lensfunc(x,y):
        return yfunc(invzt(x,y)[0],invzt(x,y)[1])

    return lensfunc
    
################## CURL UTIL    
def make_curl_func(lensfnc,dx):
    def curl_func(x1,x2):
        dydx1 = (-lensfnc(x1+2.*dx,x2) + 8*lensfnc(x1+dx,x2) - 8*lensfnc(x1-dx,x2) + lensfnc(x1-2.*dx,x2))/(12.*dx)
        dydx2 = (-lensfnc(x1,x2+2.*dx) + 8*lensfnc(x1,x2+dx) - 8*lensfnc(x1,x2-dx) + lensfnc(x1,x2-2.*dx))/(12.*dx)
        return dydx1[1] - dydx2[0]
    return curl_func
    
    
################## EIGENCURVE UTILS

def geteigen(hes):
    eigenval, eigenvec = np.linalg.eig(hes)
    eigenvec[:,0] = -eigenvec[:,0]
    return eigenval, eigenvec
    
def eigen_from_pos(pos1,pos2,kwargs):
    heslens = lens_model.hessian(pos1,pos2,kwargs)
    hes = np.array([[1.,0.],[0.,1.]]) - np.array([[heslens[0],heslens[1]],[heslens[2],heslens[3]]])
    eigenval, eigenvec = geteigen(hes)
    
def get_theta_from_vec(eigenvec):
    thetacan = np.arctan2(eigenvec[1], eigenvec[0])
    if thetacan < -np.pi/2:
        thetacan += np.pi
    elif thetacan > np.pi/2:
        thetacan += -np.pi
    return thetacan
    
def get_lambda_1_lambda_2_theta(x10,x20,size,Npix,lens_model,kwargs):   
    x1val = np.linspace(x10-size ,x10+size ,Npix)
    x2val = np.linspace(x20-size ,x20+size,Npix)
    
    mag = np.zeros([Npix,Npix])
    
    for i in range(Npix):
        for j in range(Npix):
            mag[i,j] = lens_model.magnification(x1val[i], x2val[j], kwargs)    

    lambda_1 = np.zeros([Npix,Npix])
    lambda_2 = np.zeros([Npix,Npix])
    theta = np.zeros([Npix,Npix])

    for i in range(Npix):
        for j in range(Npix):
            x1 = x1val[i]
            x2 = x2val[j]
            eigenvald, eigenvecd = eigen_from_pos(x1,x2,kwargs)
            
            if np.abs(eigenvald[0]) <= np.abs(eigenvald[1]):
                lambda_1[i,j] = eigenvald[0]
                lambda_2[i,j] = eigenvald[1]
                thetacan = get_theta_from_vec(eigenvec[0])
                theta[i,j] = thetacan
            else:
                lambda_1[i,j] = eigenvald[1]
                lambda_2[i,j] = eigenvald[0]
                thetacan = get_theta_from_vec(eigenvec[1])
                theta[i,j] = thetacan
    
    return lambda_1,lambda_2,theta,mag
    

    
def get_primary_curve(d1,d2,lens_model,kwargs,dx,Niter):
    ### starting from (d1,d2) iterates to find the primary curve
    
    curve1 = np.zeros(2*Niter+1)
    curve2 = np.zeros(2*Niter+1)

    lambda_1 = np.zeros(2*Niter+1)
    lambda_2 = np.zeros(2*Niter+1)
    
    Ffunc = np.zeros(2*Niter+1)
    Gfunc = np.zeros(2*Niter+1)

    curve1[Niter] = d1
    curve2[Niter] = d2

    x1_fiter = d1
    x2_fiter = d2

    x1_biter = d1
    x2_biter = d2

    eigenvald, eigenvecd = eigen_from_pos(d1,d2,kwargs)
    
    if np.abs(eigenvald[0]) <= np.abs(eigenvald[1]):
        lambda_1[Niter] = eigenvald[0]
        lambda_2[Niter] = eigenvald[1]
        vecalign = eigenvecd[0]
    else:
        lambda_1[Niter] = eigenvald[1]
        lambda_2[Niter] = eigenvald[0]
        vecalign = eigenvecd[1]

    for i in range(Niter+1):
        eigenvalf, eigenvecf = eigen_from_pos(x1_fiter,x2_fiter,kwargs)
        ds = 1e-5    
        if np.abs(eigenvalf[0]) <= np.abs(eigenvalf[1]):
            stepvecf = dx*eigenvecf[0]*np.sign(np.dot(vecalign,eigenvecf[0]))
            lambda_1[Niter+i] = eigenvalf[0]
            lambda_2[Niter+i] = eigenvalf[1]
            
            
            ## higher order derivatives w/r/to s
            orthostep = dx*eigenvecf[1]*np.sign(np.dot(vecalign,eigenvecf[0]))
            
            x1ortho = x1_fiter + orthostep[0]
            
            eigenvalf_plus, eigenvecf_plus = eigen_from_pos(x1_fiter + ds*orthostep[0],x2_fiter + ds*orthostep[1],kwargs)
            eigenvalf_neg, eigenvecf_neg = eigen_from_pos(x1_fiter - ds*orthostep[0],x2_fiter - ds*orthostep[1],kwargs)
            
            ## Implement F, G, H, I calculation here
            
            
        else:
            stepvecf = dx*eigenvecf[1]*np.sign(np.dot(vecalign,eigenvecf[0]))
            lambda_1[Niter+i] = eigenvalf[1]
            lambda_2[Niter+i] = eigenvalf[0]
            
            
            ## higher order derivatives w/r/to s
            orthostep = dx*eigenvecf[0]*np.sign(np.dot(vecalign,eigenvecf[0]))
            
      
            eigenvalf_plus, eigenvecf_plus = eigen_from_pos(x1_fiter + ds*orthostep[0],x2_fiter + ds*orthostep[1],kwargs)
            eigenvalf_neg, eigenvecf_neg = eigen_from_pos(x1_fiter - ds*orthostep[0],x2_fiter - ds*orthostep[1],kwargs)
            
            

        x1_fiter += stepvecf[0]
        x2_fiter += stepvecf[1]

        curve1[Niter+i] = x1_fiter
        curve2[Niter+i] = x2_fiter

        eigenvalb, eigenvecb = eigen_from_pos(x1_biter,x2_biter,kwargs)

        if np.abs(eigenvalb[0]) <= np.abs(eigenvalb[1]):
            stepvecb = -dx*eigenvecb[0]*np.sign(np.dot(vecalign,eigenvecb[0]))
            lambda_1[Niter-i] = eigenvalb[0]
            lambda_2[Niter-i] = eigenvalb[1]
        else:
            stepvecb = -dx*eigenvecb[1]*np.sign(np.dot(vecalign,eigenvecb[0]))
            lambda_1[Niter-i] = eigenvalb[1]
            lambda_2[Niter-i] = eigenvalb[0]

        x1_biter += stepvecb[0]
        x2_biter += stepvecb[1]

        curve1[Niter-i] = x1_biter
        curve2[Niter-i] = x2_biter
    
    
    return curve1,curve2,lambda_1,lambda_2,vecalign
    
    
 
def get_image_coords(image,dPix):
    shp = np.shape(image)
    ##shp[0] should be equal to shp[1]
    ext = np.linspace(-dPix*(0.5*shp[0]-0.5),dPix*(0.5*shp[0]-0.5),shp[0])
    xgrid, ygrid = np.meshgrid(ext, ext)
    return xgrid,ygrid
    
    
def flatten_data(image,dPix,likemask):
    xgrid, ygrid = get_image_coords(image,dPix)
    dataarray = []
    coordinate_array = []
    for i in range(np.shape(likemask)[0]):
        for j in range(np.shape(likemask)[1]):
            if likemask[i,j] != 0.:
                dataarray.append(image[i,j])
                coordinate_array.append([xgrid[i,j],ygrid[i,j]])
    dataarray = np.array(dataarray)
    coordinate_array = np.array(coordinate_array)
    return dataarray,coordinate_array
def unflatten_data(flatdata,likemask):
    unflat_data = np.zeros(np.shape(likemask))
    iter = 0
    for i in range(np.shape(likemask)[0]):
        for j in range(np.shape(likemask)[1]):
            if likemask[i,j] != 0.:
                unflat_data[i,j] = flatdata[iter]
                iter += 1
    return unflat_data
    
def rechunk(array2d,nchunk):
    shp = np.shape(array2d)
    shpnew = [int(q/nchunk) for q in shp]
    arraynew = np.zeros(shpnew)
    
    for i in range(shpnew[0]):
        for j in range(shpnew[1]):
            for k in range(nchunk):
                for l in range(nchunk):
                    arraynew[i,j] += array2d[i*nchunk+k,j*nchunk+l]/(nchunk**2.) 
    return arraynew


#### TO_DO BETTER PIXEL SELECTION CODE!!!
#### DO IT SOON!
def get_tessalation_pixels(flatdata,flatcoords,Ntess,Nskip):
    sorted_args = np.argsort(-flatdata)
    sorted_dat = flatdata[sorted_args]
    sorted_coords = flatcoords[sorted_args]

    selection = np.arange(0,Ntess*Nskip,Nskip)
    return sorted_dat[selection],sorted_coords[selection]
