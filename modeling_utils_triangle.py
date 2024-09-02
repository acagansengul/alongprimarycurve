import random
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from scipy.optimize import nnls
import func_utils as fu

nums = list(range(0, 1000)) # list of integers from 1 to 99
                           # adjust this boundaries to fit your needs
random.shuffle(nums)



def make_source_cells(imgx,imgy,yfunc):
    source = yfunc(imgx,imgy)
    sourcex = source[0]
    sourcey = source[1]
    return sourcex,sourcey

def get_amplitude_effect(posvals,pos):
    pos1,pos2,pos3 = posvals[0],posvals[1],posvals[2]
    x_1,y_1 = pos1[0],pos1[1]
    x_2,y_2 = pos2[0],pos2[1]
    x_3,y_3 = pos3[0],pos3[1]
    x,y = pos[0],pos[1]

    c1 = ((y_2-y_3)*(x-x_3) - (x_2-x_3)*(y-y_3))/((x_1-x_3)*(y_2-y_3) - (x_2-x_3)*(y_1-y_3))
    c2 = ((y_3-y_1)*(x-x_1) - (x_3-x_1)*(y-y_1))/((x_2-x_1)*(y_3-y_1) - (x_3-x_1)*(y_2-y_1))
    c3 = ((y_1-y_2)*(x-x_2) - (x_1-x_2)*(y-y_2))/((x_3-x_2)*(y_1-y_2) - (x_1-x_2)*(y_3-y_2))

    return np.array([c1,c2,c3])

def get_triangle_equation(Avals,posvals):
    A_1,A_2,A_3 = Avals[0],Avals[1],Avals[2]
    
    def trifunc(x,y):
        pos = np.array([x,y])
        carr = get_amplitude_effect(posvals,pos)
        return carr[0]*Avals[0] + carr[1]*Avals[1] + carr[2]*Avals[2]

    return trifunc


def map_source_intensity(yfunc,source_KDtree):
    ## returns a function point that (t,s) gets mapped to
    def int_map(x,y):
        sx,sy = yfunc(x,y)
        d, i = source_KDtree.query((sx,sy))
        return i
    return int_map
    
def map_source_intensity_triangle(yfunc,delau):
    ## returns a function point that (t,s) gets mapped to
    def int_map(x,y):
        sx,sy = yfunc(x,y)
        testpoint =np.array([sx,sy])
        where = delau.find_simplex(testpoint)
        if where == -1:
            triwithin = np.array([-1,-1,-1])
        else:
            triwithin = delau.simplices[where]
        return triwithin
    return int_map

def get_source_amps(mapping,data):
    Amatrix = np.matmul(mapping.T, mapping)
    Bmatrix = np.matmul(mapping.T, data)        
    invA = np.linalg.inv(Amatrix)
    amp = np.matmul(invA, Bmatrix)
    modelpred = np.matmul(mapping, amp)
    return amp,modelpred\
    
def get_source_amps2(mapping,data):
    Amatrix = np.matmul(mapping.T, mapping)
    Bmatrix = np.matmul(mapping.T, data)       
    #invA = np.linalg.inv(Amatrix)
    print('correct one NNLS')
    amp = nnls(Amatrix,Bmatrix)[0]
    modelpred = np.matmul(mapping, amp)
    return amp,modelpred
    
def flatten_psf(psfarray):
    psfflat = []
    for i in range(np.shape(psfarray)[0]):
        for j in range(np.shape(psfarray)[1]):
            psfflat.append(psfarray[i,j])
    psfflat = np.array(psfflat)
    return psfflat

def find_neighbors(point,point_array,ndist,deltaPix):
    distances = np.linalg.norm(point_array - point,ord=np.inf,axis=1)
    fullid = np.arange(len(point_array[:,0]))
    indices = fullid[distances <= ndist*deltaPix]

    orientations = (np.round((point_array[indices] - point)/deltaPix))+np.array([7.,7.])
    orientations = orientations.astype(int)
    return indices,orientations


def make_psf_matrix(psf_array,flatdata,coords,deltaPix):
    ## psf array has to be size 15x15
    flatpsf = flatten_psf(psf_array)
    psf_matrix = np.zeros([len(flatdata),len(flatdata)])
    for i in range(len(flatdata)):
        indtest,orients = find_neighbors(coords[i],coords,7,deltaPix)
        if len(indtest) == 225:
            psf_matrix[i,indtest] = flatpsf
        else:
            for k in range(len(indtest)):
                psf_matrix[i,indtest[k]] = psf_array[orients[k,0],orients[k,1]]
                
    return psf_matrix
    
    
def get_8_points(coord,deltaPix):
    crd1 = coord + np.array([deltaPix/3.,0])
    crd2 = coord + np.array([deltaPix/3.,deltaPix/3.])
    crd3 = coord + np.array([0.,deltaPix/3.])
    crd4 = coord + np.array([-deltaPix/3.,deltaPix/3.])
    crd5 = coord + np.array([-deltaPix/3.,0.])
    crd6 = coord + np.array([-deltaPix/3.,-deltaPix/3.])
    crd7 = coord + np.array([0.,-deltaPix/3.])
    crd8 = coord + np.array([deltaPix/3.,-deltaPix/3.])
    return np.array([crd1,crd2,crd3,crd4,crd5,crd6,crd7,crd8])


def linearfit_eigencurve_basis_deflection_image_FAST(lens_params,flatdata,scord,flatcoords,psf_matrix,deltaPix):
    
    curve_params = lens_params['curve_params'] 
    lambda_1poly = lens_params['lambda_1poly']
    lambda_2poly = lens_params['lambda_2poly']
    Fpoly = lens_params['Fpoly']
    H0 = lens_params['H0']
    
    
    lensfunc0 = fu.make_lens_func(curve_params,lambda_1poly,lambda_2poly,Fpoly,H0)

    ######## MAKE SOURCE KDTREE
    sourcex = np.zeros(len(scord[:,0]))
    sourcey = np.zeros(len(scord[:,0]))
    for i in range(len(scord[:,0])):
        sourcex[i],sourcey[i] = make_source_cells(scord[i,0],scord[i,1],lensfunc0)
    sourcepointarray = np.array([sourcex,sourcey]).T
    delaunay = Delaunay(sourcepointarray)
    kdtree = KDTree(sourcepointarray)
    ########

    int_map_test_triangle = map_source_intensity_triangle(lensfunc0,delaunay)
    int_map_test = map_source_intensity(lensfunc0,kdtree)
    
    coeffs = np.zeros([len(flatdata),3])
    indices = np.zeros([len(flatdata),3])


    for i in range(len(flatdata)):
        index = int_map_test_triangle(flatcoords[i,0],flatcoords[i,1])
        if index[0] == -1:
            indexnearest = int_map_test(flatcoords[i,0],flatcoords[i,1])
            coeffs[i] = np.array([1.,1.,1.])
            indices[i] = np.array([indexnearest,indexnearest,indexnearest])
        else:
            source_triangle = sourcepointarray[index]
            lensedx,lensedy = lensfunc0(flatcoords[i,0],flatcoords[i,1])
            coeffs[i] = get_amplitude_effect(source_triangle,[lensedx,lensedy])
            indices[i] = index#np.array([-1,-1,-1])
    mapping_matrix0 = np.zeros([len(flatdata),np.shape(sourcepointarray)[0]])
    for i in range(len(flatdata)):
        for j in range(3):
            #if indices[i,j] != -1:
            mapping_matrix0[i,int(indices[i,j])] = coeffs[i,j]
            
    mapping_matrix = np.matmul(psf_matrix,mapping_matrix0)

#    singulartest = np.min(np.max(np.abs(mapping_matrix),axis=0))
#    if singulartest == 0.:
#        print('singular!')
#        amps,models = np.zeros(len(scord[:,0])),np.zeros(len(flatdata))
#    else:
        #amps,models = get_source_amps(mapping_matrix,flatdata)
    amps,models = get_source_amps2(mapping_matrix,flatdata)
    
    return amps,models,lensfunc0,delaunay    
    
    
def linearfit_eigencurve_basis_deflection_image(lens_params,flatdata,scord,flatcoords,psf_matrix,deltaPix):
    curve_params = lens_params['curve_params'] 
    lambda_1poly = lens_params['lambda_1poly']
    lambda_2poly = lens_params['lambda_2poly']
    Fpoly = lens_params['Fpoly']
    H0 = lens_params['H0']

    lensfunc0 = fu.make_lens_func(curve_params,lambda_1poly,lambda_2poly,Fpoly,H0)

    ######## MAKE SOURCE KDTREE
    sourcex = np.zeros(len(scord[:,0]))
    sourcey = np.zeros(len(scord[:,0]))
    for i in range(len(scord[:,0])):
        sourcex[i],sourcey[i] = make_source_cells(scord[i,0],scord[i,1],lensfunc0)
    sourcepointarray = np.array([sourcex,sourcey]).T
    delaunay = Delaunay(sourcepointarray)
    kdtree = KDTree(sourcepointarray)
    ########

    int_map_test_triangle = map_source_intensity_triangle(lensfunc0,delaunay)
    int_map_test = map_source_intensity(lensfunc0,kdtree)
    
    coeffs = np.zeros([len(flatdata),3])
    indices = np.zeros([len(flatdata),3])
    
    for i in range(len(flatdata)):
        points8 = get_8_points(flatcoords[i],deltaPix)
        coeff = 0.
        for j in range(9):
            if j == 8:
                index = int_map_test_triangle(flatcoords[i,0],flatcoords[i,1])
                if index[0] == -1:
                    indexnearest = int_map_test(flatcoords[i,0],flatcoords[i,1])
                    coeff += np.array([1.,1.,1.])/9.
                    indices[i] = np.array([indexnearest,indexnearest,indexnearest])
                else:
                    source_triangle = sourcepointarray[index]
                    lensedx,lensedy = lensfunc0(flatcoords[i,0],flatcoords[i,1])
                    coeff += get_amplitude_effect(source_triangle,[lensedx,lensedy])/9
                    indices[i] = index
            else:
                index = int_map_test_triangle(flatcoords[i,0],flatcoords[i,1])
                if index[0] == -1:
                    indexnearest = int_map_test(points8[j,0],points8[j,1])
                    coeff += np.array([1.,1.,1.])/9.
                    indices[i] = np.array([indexnearest,indexnearest,indexnearest])
                else:
                    source_triangle = sourcepointarray[index]
                    lensedx,lensedy = lensfunc0(points8[j,0],points8[j,1])
                    coeff += get_amplitude_effect(source_triangle,[lensedx,lensedy])/9
                    indices[i] = index
            coeffs[i] = coeff
        mapping_matrix0 = np.zeros([len(flatdata),np.shape(sourcepointarray)[0]])
    for i in range(len(flatdata)):
        for j in range(3):
            #if indices[i,j] != -1:
            mapping_matrix0[i,int(indices[i,j])] = coeffs[i,j]


    mapping_matrix = np.matmul(psf_matrix,mapping_matrix0)

#    singulartest = np.min(np.max(np.abs(mapping_matrix),axis=0))
#    if singulartest == 0.:
#        print('singular!')
#        amps,models = np.zeros(len(scord[:,0])),np.zeros(len(flatdata))
#    else:
        #amps,models = get_source_amps(mapping_matrix,flatdata)
    amps,models = get_source_amps2(mapping_matrix,flatdata)
    
    return amps,models,lensfunc0,delaunay
    
    
    
def visualize_delaunay_source(delaunay,amplitudes,xlow,xhigh,ylow,yhigh):
    sourcepoints = delaunay.points
    print(np.shape(sourcepoints))
    #imgsizex = np.max(sourcepoints[:,0]) - np.min(sourcepoints[:,0])
    #imgsizey = np.max(sourcepoints[:,1]) - np.min(sourcepoints[:,1])
    #size = np.max([imgsizex,imgsizey])
    #amax = np.argmax(amplitudes)
    #center = sourcepoints[amax]
    Nres = 1000
    xvals = np.linspace(xlow,xhigh,Nres)
    yvals = np.linspace(ylow,yhigh,Nres)
    
    xiter = xvals[1]-xvals[0]
    yiter = yvals[1]-yvals[0]
    
    xmin = xvals[0] - xiter/2.
    xmax = xvals[-1] + xiter/2.
    
    ymin = yvals[0] - yiter/2.
    ymax = yvals[-1] + yiter/2.   
    
    
    print(np.max(xvals),np.min(xvals))
    print(np.max(yvals),np.min(yvals))
    
    surface_brightness = np.zeros([Nres,Nres])
    for i in range(Nres):
        for j in range(Nres):
            testpoint =np.array([xvals[i],yvals[j]])
            where = delaunay.find_simplex(testpoint)
            if where == -1:
                triwithin = np.array([-1,-1,-1])
                surface_brightness[i,j] = 0.
            else:
                triwithin = delaunay.simplices[where]
                source_triangle = sourcepoints[triwithin]
                coeffs = get_amplitude_effect(source_triangle,[xvals[i],yvals[j]])
                trieq = get_triangle_equation(amplitudes[triwithin],source_triangle)
                surface_brightness[i,j] = trieq(xvals[i],yvals[j])


    return surface_brightness,xvals,yvals, xmin,xmax,ymin,ymax