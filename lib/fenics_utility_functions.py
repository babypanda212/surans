# standard imports
from fenics import *
import numpy as np

# project related imports
from lib import utility_functions as ut


def l_m(fparams, params):
    
    x     = fparams.x    # it is the height variable 
    U_g   = fparams.U_g
    
    f_c   = params.f_c
    kappa = params.kappa
    
    if params.stochastic_phi:
        phi = f_m(fparams, params) * sigmoid(fparams, params) + fparams.f_ms * (1 - sigmoid(fparams, params))
    else:
        phi = f_m(fparams, params)

    return (kappa * x[0]) / (phi + (kappa * x[0]) / lambbda(U_g, f_c))


# Cuxart 2006 table 3
def K_m(fparams, params):
    return params.alpha * l_m(fparams, params) * sqrt(fparams.k + 1E-16)


# Cuxart 2006 table 3 
def K_h(fparams, params):
    return K_m(fparams, params) / params.Pr_t


# Rodrigo 2013 Eq. 27, The TKE Generation term due to wind shear and buoyancy.
def G_E(fparams, params):
    
    T = fparams.T
    u = fparams.u
    v = fparams.v
    
    g     = params.g
    T_ref = params.T_ref
    
    return K_m(fparams, params) * S(u, v) - g/(T_ref) * K_h(fparams, params) * T.dx(0)


# Rodrigo 2013 Eq. 29, The turbulent dissipation rate eps
def eps(fparams, params):
    return  params.alpha_e * (fparams.k)**(3/2) * ( 1/l_m(fparams, params) + 1E-16 )

# this function computes the Ri number in the simulation loop. The Ri number is
# then used to solve the stochastic equation. 
def Ri_stand_alone(fparams, params, T_loc, u_loc, v_loc):
    
    S_loc = S(u_loc, v_loc)
    N_loc = N(T_loc, params)
    
    calc_Ri = abs( N_loc / (S_loc + 1E-16) )
    Ri_con  = project(conditional(gt(calc_Ri, 10.0), 10.0, calc_Ri), fparams.Q)
    
    # write out a numpy array
    return np.flipud( Ri_con.vector().get_local() )


#for debuging only
def f_m_stand_alone(Ri):
    return 1.0 + 12.0 * Ri
    
# Sorbjan 2012 after Eq. 3b. Since we did not take the sqrt(S) and sqrt(N) we
# do not compute N**2 / S**2 but N / S. We add the EPS, so we do not get division
# by zero.
def Ri(fparams, params):
    calc_Ri = abs( N(fparams.T, params) / (S(fparams.u, fparams.v) + 1E-16) )
    # limit the Ri to max value of 10 (for solver stability)
    return conditional(gt(calc_Ri, 10.0), 10.0, calc_Ri)


# Sorbjan 2012 Eq. 3b
# Gradients of the wind. For computing the Ri Number and the eddy diffusivities.
# We take the sqrt(S) later in the definition of the eddy diffusivities. 
def S(u, v):
    return u.dx(0)**2 + v.dx(0)**2 


# Sorbjan 2012 Eq. 3b
# Brunt-Vaisala frequency. For computing the Ri Number. We do not take the
# sqrt(N) to allow for negativ values.
def N(T, params):
    return params.beta * T.dx(0) 


def lambbda(U_g, f_c):
    return 3.7 * 1e-4 * U_g / f_c


# Sorbjan 2012 Eq. 5a and 5b. Empirical stability functions. They correct the
# turbulent mixing depending on the local stability.
def f_m(fparams, params):
    return 1.0 + 12.0 * Ri(fparams, params)


# as above. The heat is different from momentum
def f_h(fparams, params):
    return 1.0 + 12.0 * Ri(fparams, params)


def sigmoid(fparams, params):
    
    z   = fparams.x[0]
    z_l = params.z_l
    
    k = 0.1
    return  1 / (1 + exp(-k*(z - z_l)))

def weak_formulation(fparams, params, u_n, v_n, T_n, k_n):
    u, v, T, k, w_u, w_v, w_T, w_k, x, U_g, V_g = unroll(fparams)
        
    Dt  = Constant(params.dt)
    Fc  = Constant(params.f_c)
    tau = Constant(params.tau)
    H  = params.H
    
    # # Define variational problem
    #---------- Velocity--u-comp------------------
    F_u = - Fc * (v - V_g) * w_u * dx  \
          + K_m(fparams, params) * dot(u.dx(0), w_u.dx(0)) * dx \
          + ((u - u_n) / Dt) * w_u * dx \
          + (u - U_g) / tau * w_u * dx    
    
    #---------- Velocity--v-comp------------------
    F_v = + Fc * (u - U_g) * w_v * dx  \
          + K_m(fparams, params) * dot(v.dx(0), w_v.dx(0)) * dx \
          + ((v - v_n) / Dt) * w_v * dx \
          + (v - V_g ) / tau * w_v * dx
         
    #--------------Temperature--------------------
    F_T = + K_h(fparams, params) * dot(T.dx(0), w_T.dx(0)) * dx \
          - K_h(fparams, params) * params.gamma * w_T * ds \
          + ((T - T_n) / Dt) * w_T * dx \
    
    #------------------TKE------------------------
    F_k = + K_m(fparams, params) * dot(k.dx(0), w_k.dx(0)) * dx \
          - G_E(fparams, params) * w_k * dx\
          + eps(fparams, params) * w_k * dx\
          + ((k - k_n) / Dt) * w_k * dx 
    
    F = F_u + F_v + F_T + F_k
    
    return F


def setup_fenics_variables(fparams, mesh):
    
    fparams.W = VectorFunctionSpace(mesh, 'CG', 1, dim=4)

    # Define test functions
    fparams.w_u, fparams.w_v, fparams.w_T, fparams.w_k = TestFunctions(fparams.W)
    
    # Split system functions to access components
    fparams.uvTk = Function(fparams.W)
    fparams.u, fparams.v, fparams.T, fparams.k = split(fparams.uvTk)
    
    #height "z"
    fparams.x = SpatialCoordinate(mesh)
    fparams.z = mesh.coordinates()        # Grid of the simulation domain.
    
    # Function space for projection. For writing out variables
    fparams.Q = FunctionSpace(mesh, "CG", 1)

    # Function for the mixing lengt
    fparams.l0 = Function(fparams.Q)
    fparams.l0 = Constant(0.01)
    
    return fparams


def prepare_fenics_solver(fparams, F):
    
    set_log_level(LogLevel.WARNING)  # supress fenics output

    J = derivative(F, fparams.uvTk)
      
    problem = NonlinearVariationalProblem(F, fparams.uvTk, fparams.bc, J)
    
    solver = NonlinearVariationalSolver(problem)
    info(solver.parameters,True)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
    solver.parameters["newton_solver"]['absolute_tolerance'] = 1E-6
    solver.parameters["newton_solver"]['relative_tolerance'] = 1E-6
    
    return solver
    

def unroll(fparams):
    
    u = fparams.u
    v = fparams.v
    T = fparams.T
    k = fparams.k

    w_u = fparams.w_u
    w_v = fparams.w_v
    w_T = fparams.w_T
    w_k = fparams.w_k
    
    x = fparams.x
    
    U_g = fparams.U_g
    V_g = fparams.V_g
    
    return u, v, T, k, w_u, w_v, w_T, w_k, x, U_g, V_g
    
