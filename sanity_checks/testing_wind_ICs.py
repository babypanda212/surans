import matplotlib.pyplot as plt
import numpy as np
# project-related imports
from lib import initial_and_boundary_conditions as ic
from lib import fenics_utility_functions as fut 
from lib import space_discretization as sd
from lib import write_out_functions as wo
from lib import utility_functions as ut
from lib import stochastic_model as sm
from lib import params_class_base as pc

# Initialize parameters
params, fparams, output = pc.initialize_project_variables()

U_g = params.U_top
z0 = params.z0

# Create grid and extract mesh coordinates
mesh, params = sd.create_grid(params, 'power', show=False)
z = mesh.coordinates()

# Compute initial conditions
u_init_new = ic.initial_u0(z, U_g, z0, params)
u_init_gabls = np.ones(z.shape[0])*U_g

def initial_u0_surans(z, U_g, z0, params):
    tau_w = 0.5 * 0.004 * params.rho * U_g ** 2
    u_star_ini = np.sqrt(tau_w / params.rho)
    return u_star_ini / params.kappa * np.log(z / z0)

u_init_surans = initial_u0_surans(z, U_g, z0, params)

# Plot the data
plt.plot(u_init_surans, z, label='u_init_surans',marker='o')
plt.plot(u_init_gabls, z, label='u_init')
plt.plot(u_init_new, z, label='u_init_new',marker='x')
plt.xlabel('Velocity')
plt.ylabel('Height (z)')
plt.title('Comparison of Initial Conditions')
plt.legend()
plt.show()
