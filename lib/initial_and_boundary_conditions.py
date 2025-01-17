# standard imports
import numpy as np
from fenics import *
from scipy.interpolate import CubicSpline

# project related imports
from lib import fenics_utility_functions as fut 
from lib import utility_functions as ut


def def_initial_cnditions(Q, mesh, params):
    
    z0           = params.z0          # roughness length in meter
    Nz           = params.Nz          # number of point/ domain resolution
    H            = params.H           # domain height in meters
    U_top        = params.U_top       # u geostrophic wind
    initCondStr  = params.initCondStr # name of the file  
    
    load_ini_cond = params.load_ini_cond  # bool type; load existing initial condition
    
    u_n = Function(Q)
    v_n = Function(Q)
    T_n = Function(Q)
    k_n = Function(Q)
    
    z = mesh.coordinates()
    
    t1 = Function(Q)
    if load_ini_cond:
        t1.vector().set_local(np.flipud(np.load(initCondStr + '_u.npy')))
        u_n = project(t1, Q)
        
        t1.vector().set_local(np.flipud(np.load(initCondStr + '_v.npy')))
        v_n = project(t1, Q)
        
        t1.vector().set_local(np.flipud(np.load(initCondStr + '_T.npy')))
        T_n = project(t1, Q)
        
        t1.vector().set_local(np.flipud(np.load(initCondStr + '_k.npy')))
        k_n = project(t1, Q)
    
    else:
        t1.vector().set_local(np.flipud(initial_u0(z, U_top, z0, params)))
        u_n = project(t1, Q)
        
        t1.vector().set_local(np.flipud(0*np.ones(Nz)))
        v_n = project(t1, Q)
        
        t1.vector().set_local(np.flipud(initial_T0(z, params.Tg_n, 100)))
        T_n = project(t1, Q)
        
        t1.vector().set_local(np.flipud(initial_k0(z)))
        k_n = project(t1, Q)

        # t1.vector().set_local(np.flipud(initial_k0(params, z, U_top, z0, H)+0.01))
        # k_n = project(t1, Q)

        
    
    #import matplotlib.pyplot as plt
    #plt.plot((initial_u0(z, U_top, z0, params)), z)
    #plot(project(k_n,Q))
    return u_n, v_n, T_n, k_n


def def_boundary_conditions(fparams, params):
    
    z0          = params.z0          # roughness length in meter
    H           = params.H            # domain height in meters
    U_top       = params.U_top       # u geostrophic wind
    initCondStr = params.initCondStr # name of the file 
    
    load_ini_cond = params.load_ini_cond  # bool type; load existing initial condition
    
    V           = fparams.W  # fenics variable; the vector function space
    
    ground = 'near(x[0],' + str(z0) +',1E-6)'
    top    = 'near(x[0],' + str(H) + ',1E-6)'
    
    if load_ini_cond:
        u_ini = np.load(initCondStr + '_u.npy')
        v_ini = np.load(initCondStr + '_v.npy')
        T_ini = np.load(initCondStr + '_T.npy')
        k_ini = np.load(initCondStr + '_k.npy')
        
        u_D_low = Expression('value', degree=0, value=u_ini[0])
        u_D_top = Expression('value', degree=0, value=u_ini[-1])
        
        v_D_low = Expression('value', degree=0, value=u_ini[0])
        T_D_low = Expression('value', degree=0, value=T_ini[0])
        
        k_D_low = Expression('value', degree=0, value=k_ini[0])
        k_D_top = Expression('value', degree=0, value=k_ini[-1])
        
        # velocity u component
        bcu_ground = DirichletBC(V.sub(0), u_D_low, ground)
        bcu_top    = DirichletBC(V.sub(0), u_D_top, top)
        
        # velocity v component
        bcv_ground = DirichletBC(V.sub(1), v_D_low, ground)
        bcv_top    = DirichletBC(V.sub(1), 0.0    , top)
        
        # Temperature
        bcT_ground = DirichletBC(V.sub(2), T_D_low, ground)
        
        # TKE
        bck_ground = DirichletBC(V.sub(3), k_D_low, ground)
        bck_top    = DirichletBC(V.sub(3), k_D_top, top)
        
        bc = [bcu_ground, bcv_ground, bcT_ground, bck_ground, bcv_top]
        
        Tg_n = T_ini[0]

    else:
    
        u_D_low = Expression('value', degree=0, value=0.0)
        v_D_low = Expression('value', degree=0, value=0.0)
        T_D_low = Expression('value', degree=0, value=params.Tg_n)
        k_D_low = Expression('value', degree=0, value=initial_k0(z0))
        
        #TODO: Find upper boundary condition for TKE. Placeholder TKE(H)=0
        k_D_high = Expression('value', degree=0, value=0.0001)
            
        # velocity u component
        bcu_ground = DirichletBC(V.sub(0), u_D_low, ground)
        bcu_top    = DirichletBC(V.sub(0), U_top    , top)
        
        # velocity v component
        bcv_ground = DirichletBC(V.sub(1), v_D_low, ground)
        bcv_top    = DirichletBC(V.sub(1), 0.0    , top)
        
        # Temperature
        bcT_ground = DirichletBC(V.sub(2), T_D_low, ground)
        
        # TKE
        bck_ground = DirichletBC(V.sub(3), k_D_low, ground)
        bck_top    = DirichletBC(V.sub(3), k_D_high, top)
        
        bc = [bcu_ground, bcv_ground, bcT_ground, bck_ground, bcv_top]
        
        Tg_n = params.Tg_n
    
    
    # writing out the fenics parameters
    fparams.bc      = bc           # list of boundary conditions. Will be used in the FEM formulation
    fparams.T_D_low = T_D_low      # Temperature. Fenics expression is used to control the value within the main loop solution
    fparams.k_D_low = k_D_low      # TKE.         Fenics expression is used to control the value within the main loop solution
    
    fparams.U_g = Expression('value', degree=0, value=params.U_top) # Geostrofic wind; added here to control in in the main loop
    fparams.V_g = Constant(params.V_top)                            # Geostrofic wind; added here to control in in the main loop
    
    # writing out normal parameters
    params.Tg_n = Tg_n    # The value of the Temperature at the ground.
    
    q1 = Constant(1.0)
    fparams.f_ms = Expression("value", value=q1, degree=0)
    
    
    return fparams, params


# def initial_u0(z, U_g, z0, params):
#     tau_w = 0.5 * 0.004 * params.rho * U_g ** 2
#     u_star_ini = np.sqrt(tau_w / params.rho)
#     return u_star_ini / params.kappa * np.log(z / z0)

def initial_u0(z, U_g, z0, params):
    tau_w = 0.5 * 0.004 * params.rho * U_g ** 2
    u_star_ini = np.sqrt(tau_w / params.rho)
    k = 0.01  # Adjust for steepness
    initial_u0 = U_g * (1 - (np.exp(-k * z)))
    # initial_u0[:1] = 0 
    return initial_u0


def initial_T0(z, T_ground, cut_height):
    gamma = 0.01  # K/m
    T0 = (z - cut_height) * gamma + T_ground
    ind1 = int(np.abs(z - cut_height).argmin())
    T0[0:ind1 + 1] = T_ground
    return T0


# def initial_k0(params, z, U_g, z0, H, k_at_H=0.0, _f_m=0.1):
#     # tau_w = 0.5 * 0.004 * params.rho * U_g ** 2
#     # u_star_ini = np.sqrt(tau_w / params.rho)
#     # k_at_z0 = u_star_ini**2/np.sqrt(_f_m)
#     # func = lambda z, a, b: a*np.log(z) + b
#     # a = (k_at_H - k_at_z0)/(np.log(H) - np.log(z0))
#     # b = k_at_z0 - a*np.log(z0)
#     return func(z, a, b)

def initial_k0(z, a=0.4, b=250, min_k=1e-4):
    initial_k0 = a * ((1 - (z / b)) ** 3)
    
    # Threshold k0 to ensure all values are >= min_k
    initial_k0 = np.maximum(initial_k0, min_k)
    
    return initial_k0

def calculate_cfl(z, initial_u0, dt):
    z = z.flatten() if len(z.shape) > 1 else z
    print("Flattened z:", z)

    z = np.flip(z)
    
    # Ensure dz is calculated
    dz = np.diff(z)
    dz = np.append(dz, dz[-1])
    
    # Compute CFL values
    try:
        cfl_values = (initial_u0.vector().get_local() * dt) / dz
        print("CFL values calculated:", cfl_values)
    except Exception as e:
        print(f"Error calculating CFL values: {e}")
        raise
    return cfl_values
