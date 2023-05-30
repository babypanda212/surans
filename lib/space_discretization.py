# standard imports
import matplotlib.pyplot as plt
from tqdm import tqdm
from fenics import *
import numpy as np
import h5py

# project related imports
from lib import utility_functions as ut

#-----------------------------------------------------------------------------
def create_grid(params, grid_type, show=False):
    
    z0 = params.z0  # roughness length
    H  = params.H   # domain height in meters 
    Nz = params.Nz  # number of point/ domain resolution

    grid = []
    
    if grid_type == 'power':
        grid = power_grid(z0, H, Nz)
    
    if grid_type == 'log':
        grid = log_grid(z0, H, Nz)
    
    if grid_type == 'log_lin':
        grid = log_lin_grid(z0, H, Nz)
    
    if grid_type == 'linear':
        grid = lin_grid(z0, H, Nz)
        
    if grid_type == 'for_tests':
        grid = grid_for_tests(z0, H, Nz)
    
    if show:
        plt.figure()
        plt.title('Grid spacing')
        plt.plot( grid, 'o')
        plt.xlabel('n points')
        plt.ylabel('space')
        plt.show()
    
    
    if grid_type == 'for_tests':
        grid.shape = (Nz+1,1) 
        # define fenics mesh object
        mesh    = IntervalMesh( Nz, z0, H ) 
    else:
        grid.shape = (Nz,1) 
        # define fenics mesh object
        mesh    = IntervalMesh( Nz - 1 , z0, H ) 
    
    # set new mesh
    X       = mesh.coordinates()        
    X[:]    = grid
    
    # Is the maximum allowed grid size for the stochastic grid
    dh  = grid[1] - grid[0]

    # What is the physical height of the stochastic domain?
    # Now for that we need to get the resolved value by the regular grid
    
    # we set it heristically to doble the size of the blending eight.
    Hs_def = params.s_dom_ext * params.z_l 
    params.Hs_ind = ut.find_nearest(grid, Hs_def)
    Hs_reg_grid = grid[params.Hs_ind]
    Hs = Hs_reg_grid - params.z0
    
    # what is the minimum number of points to resolve "dh" in the stochastic domain?
    Ns_n = int(np.ceil(Hs / dh))  #number of points
    
    #  Fourier works best with 2^n girds, hence we need the next 2^n integer 
    params.Ns_n = Ns_n #int(np.power(2, np.ceil(np.log(Ns_n)/np.log(2)))) #number of points
    
    # For the stochastic grid we need the number of points and the physical height of the stochastic domain.
    # The height of the stochastic domain is without the "z0". So, we put the
    # following two variables in to the params struct.
    # Ns_n
    # Hs
    params.Hs = Hs 
    
    print(params.Ns_n)
    # We also need the new delta z in the stochastic domain. Per constrauction it should be < dz of the regular grid
    params.dz_s = Hs / params.Ns_n
    
    return mesh, params
#-----------------------------------------------------------------------------


def power_grid(z0, H, Nz):
    lb = z0 ** (1/3)
    rb = H ** (1/3)
    space = np.linspace(lb, rb, Nz)
    return space**3 

def log_grid(z0, H, Nz):
    return np.logspace(np.log10(z0), np.log10(H), Nz) 

def log_lin_grid(z0, H, Nz):
    b0 = 2.5
    return (np.logspace(np.log10(z0), np.log10(H), Nz)  + np.linspace(z0, H, Nz) ) / 2

def lin_grid(z0, H, Nz):
    return np.linspace(z0, H, Nz)

def grid_for_tests(z0, H, Nz):
    dz = (H - z0) / (Nz)
    return np.arange(z0, H + dz, dz)

