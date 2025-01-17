# standard imports
from dataclasses import dataclass
import matplotlib.pyplot as plt
from numpy.random import rand
from tqdm import tqdm
from fenics import *
import numpy as np
import runpy
import h5py
import os

# project related imports
from lib import initial_and_boundary_conditions as ic
from lib import fenics_utility_functions as fut 
from lib import space_discretization as sd
from lib import write_out_functions as wo
from lib import utility_functions as ut
from lib import stochastic_model as sm
from lib import params_class_base as pc


def solve_turb_model(fparams, params, output):
    
    # create mesh; options: 'power' , 'log', 'log_lin'
    mesh, params = sd.create_grid(params, 'power', show=False)
    
    # define variables to use the fenics lib
    fparams = fut.setup_fenics_variables(fparams, mesh)
    
    # define boundary conditions
    fparams, params = ic.def_boundary_conditions(fparams, params)
    
    # define initial profiles
    u_n, v_n, T_n, k_n = ic.def_initial_cnditions(fparams.Q, mesh, params)

    # Create a vertical grid for plotting
    z = np.flipud(mesh.coordinates())  # Flip the vertical grid
    
    # debugging attempts
    cfl_values = ic.calculate_cfl(z,u_n,params.dt)
    
    print("Initial u:", u_n.vector().get_local())
    # print("Initial TKE", k_n.vector().get_local())


    # Plot each profile
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(u_n.vector().get_local(), z)
    plt.xlabel("u (m/s)")
    plt.ylabel("Height (m)")
    plt.title("Initial Velocity u")

    plt.subplot(2, 2, 2)
    plt.plot(v_n.vector().get_local(), z)
    plt.xlabel("v (m/s)")
    plt.ylabel("Height (m)")
    plt.title("Initial Velocity v")

    plt.subplot(2, 2, 3)
    plt.plot(T_n.vector().get_local(), z)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Height (m)")
    plt.title("Initial Temperature")

    plt.subplot(2, 2, 4)
    plt.plot(k_n.vector().get_local(), z)
    plt.xlabel("TKE (m²/s²)")
    plt.ylabel("Height (m)")
    plt.title("Initial Turbulent Kinetic Energy")

    plt.tight_layout()
    plt.show()

    # setup the weak formulation of the equations
    F = fut.weak_formulation(fparams, params, u_n, v_n, T_n, k_n)

    print("Computed weak formulation fo the equations")
    
    # stochastic solver
    stoch_solver, params = sm.initialize_SDEsolver(params)
    
    # create the variables to write output
    output = wo.initialize(output, params)
    
    # define the solver and its parameters
    solver = fut.prepare_fenics_solver(fparams, F)
    
    # solve the system
    output = ut.solution_loop(solver, stoch_solver, params, output, fparams, u_n, v_n, T_n, k_n)
    
    # write the solution to h5 file
    wo.save_solution(output, params, fparams)