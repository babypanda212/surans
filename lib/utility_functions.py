# standard imports
import matplotlib.pyplot as plt
from scipy import interpolate as scInt
from tqdm import tqdm
from fenics import *
import numpy as np
import h5py

# project related imports
from lib import fenics_utility_functions as fut 
from lib import surface_energy_balance as seb
from lib import write_out_functions as wo
from lib import utility_functions as ut
from lib import stochastic_model as sm

def solution_loop(solver, stoch_solver, params, output, fparams, u_n, v_n, T_n, k_n):

    Tg_n    = params.Tg_n
    
    T_D_low = fparams.T_D_low
    k_D_low = fparams.k_D_low
    U_g     = fparams.U_g

    z_ind   = find_nearest(fparams.z, params.z_l) 
    
    # refactor in the future
    # stochastic model parameters ---------------------------------------------
    taun = 10
    dtau = params.dt / taun  # smaller time step for stable solution of the SDE
    flag = False # controlling efficient sampling. User should not change it
    v_sqrt = np.sqrt(dtau) # calculate the sqrt outside the loop to speed up
    
    phi0          = np.ones(params.Ns_n) # initialeze the solution variable
    phi1_reg_grid = np.ones(params.Nz)
    #--------------------------------------------------------------------------
    print('Solving ... ')
    t = 0 # used for control
    
    i_w = 0 # index for writing
    for i in tqdm(range(params.SimEnd)):
        try:
            solver.solve()
        except:
            print("\n Solver crashed...")
            break
        
        # get variables to export
        us, vs, Ts, ks = fparams.uvTk.split(deepcopy=True)
        u_n.assign(us)
        v_n.assign(vs)
        T_n.assign(Ts)
        k_n.assign(ks)
        
        # control the minimum tke level to prevent sqrt(tke)
        k_n = set_minimum_tke_level(params, ks, k_n)
        
        #Stochastic update goes here below
        #----------------------------------------------------------------------
        if params.stochastic_phi:
            # numpy array. Log() is not taken
            Ri_1 = np.log10(np.abs(fut.Ri_stand_alone(fparams, params, Ts, us, vs)) + 1e-5)
            
            # limit Ri = 10 wher the stochastic model is valid. note log to base 10
            Ri_1[Ri_1 > 1.0] = 1.0 
            # we need to interpolate from fine PDE grid to coarse stochastic grid
            
            # set the Ri above the stochastic layer to neutra, so submesoscale is ont generated there.Since correlation can still bring them down
            #Ri_1[z_ind:] = -5.0
            
            # plt.plot(Ri_1, fparams.z)
            f = scInt.interp1d(fparams.z[:,0], Ri_1, fill_value='extrapolate')
            Ri_stoch_grid = f(params.s_grid)[:,0]
            # Ri_stoch_grid = np.copy(Ri_1)
            
            # plt.plot(Ri_1, fparams.z[:,0], "o")
            # plt.plot(Ri_stoch_grid, params.s_grid, "o")
            # plt.yscale('log')
            a1 = sm.A1(Ri_stoch_grid)
            a2 = 10**sm.A2(Ri_stoch_grid)
            a3 = 10**sm.A3(Ri_stoch_grid, params)
            #lz = sm.l_s(2.0)

            
            for j in range(taun):
                stoch_solver.evolve(phi0, dtau, a1, a2, a3, flag, v_sqrt, params.lz)
                phi1 = stoch_solver.getState()
                phi0 = np.copy(phi1)
            

            # we need to interpolate from fine stochastic grid to finer PDE grid     
            f = scInt.interp1d(params.s_grid[:,0], phi1,fill_value='extrapolate')
            phi1_reg_grid[0:params.Hs_ind] = f(fparams.z[0:params.Hs_ind,0])
            # phi1_fine = np.copy(phi1)
            
            # print(np.shape(phi1))
            # plt.plot(phi1_reg_grid, fparams.z[:,0] )
            # #plt.plot(phi1,params.s_grid[:,0])
            # raise SystemError
            
            # change the value of the stochastic stab correction
            phi1_fine_pr = project(fparams.f_ms, fparams.Q)
            phi1_fine_pr.vector().set_local(np.flipud(phi1_reg_grid))
            fparams.f_ms.value = phi1_fine_pr
            
            # Diagnostics  variable
            #f_mc = fut.f_m_stand_alone(np.abs(fut.Ri_stand_alone(fparams, params, Ts, us, vs)) + 1e-5)
            #sig1 = np.flipud(project(fut.sigmoid(fparams, params), fparams.Q).vector().get_local())
            #fm_m = f_mc * sig1 + phi1_fine * (1 - sig1)
        else:
            phi1_fine_pr = project(fparams.f_ms, fparams.Q)
               
            
        #----------------------------------------------------------------------
        
        if (i)%params.Save_dt==0:
            # We first write out the variables, since the eddiy diffusivities gona be used anyway to update the boundary conditions.
            output = wo.save_current_result(output, params, fparams, i_w, us, vs, Ts, ks, phi1_fine_pr)
            i_w += 1

        
        # calc some variables
        u_now, v_now, Kh_now = wo.calc_variables_np(params, fparams, us, vs)
        
        # solve temperature at the ground
        Tg = seb.RHS_surf_balance_euler(Tg_n, fparams, params, Ts, ks, u_now, v_now, Kh_now)      
        
        # update temperature for the ODE
        Tg_n = np.copy(Tg)
        
        # update temperature for the PDE
        T_D_low.value = np.copy(Tg)
        
        # ugdate boundary conditions
        seb.update_tke_at_the_surface(fparams, params, u_now, v_now )
    
        
        # if t > (12 * 3600) and t < (18 * 3600):
        #     fparams.U_g.value = fparams.U_g.value + 1 / (3600/params.dt)
        
        # if t > (5 * 3600) and t < (6 * 3600):
        #     params.R_n = params.R_n - 40 / (3600/params.dt)
        
        # if t > (24 * 3600) and t < (25 * 3600):
        #     params.R_n = params.R_n + 40 / (3600/params.dt)
            
        i += 1
        t += params.dt


    return output
# =============================================================================


#============================ Common Functions================================
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    return np.abs(a - a0).argmin()


def set_minimum_tke_level(params, ks, k_n):
    
    # "ks" is the current solution
    # "k_n" the initial value for the next iteration
    
    # limiting the value by converting to numpy. Hmm.. there is must be a better way.
    ks_array = ks.vector().get_local()
    
    # set back a to low value
    ks_array[ks_array < params.min_tke] = params.min_tke
    
    # cast numpy to fenics variable
    ks.vector().set_local(ks_array)
    
    #update the value for the next simulation run
    k_n.assign(ks)
    
    return k_n
    









