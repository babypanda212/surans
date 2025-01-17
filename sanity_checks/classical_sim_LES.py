import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Initialize dictionary to store LES data
les_data = {}
sim_names = ['CORA', 'CSU', 'IMUK', 'LLNL', 'MO', 'NCAR', 'NERSC', 'UIB']

# Define the number of variables for A9 and B9 datasets
num_variables_A = 4  # Variables in A9 (Height, x-velocity, y-velocity, Potential temperature)
num_variables_B = 7  # Variables in B9 (Height, variances, TKE, etc.)

# Function to safely convert strings to float, handling invalid values
def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return np.nan  # Replace invalid values with NaN

# Load LES data from .dat files
for name in sim_names:
    les_data[name] = {}

    # Read {name}_A9_128.dat
    with open(f'gabls_les/res_3.125m/{name}/{name}_A9_128.dat') as f:
        metadata = f.readline().strip()
        les_data[name]['A_metadata'] = metadata
        num_elements = int(f.readline().strip())
        les_data[name]['A_num_elements'] = num_elements

        A_data = []
        for line in f:
            values = [safe_float_conversion(x) for x in line.split()]
            A_data.extend(values)
        les_data[name]['A_data'] = np.array(A_data).reshape(num_variables_A, num_elements)

    # Read {name}_B9_128.dat
    with open(f'gabls_les/res_3.125m/{name}/{name}_B9_128.dat') as f:
        metadata = f.readline().strip()
        les_data[name]['B_metadata'] = metadata
        num_elements = int(f.readline().strip())
        les_data[name]['B_num_elements'] = num_elements

        B_data = []
        for line in f:
            values = [safe_float_conversion(x) for x in line.split()]
            B_data.extend(values)
        les_data[name]['B_data'] = np.array(B_data).reshape(num_variables_B, num_elements)

# Load .h5 file data
solStringName = 'ini_wind'
file = h5py.File(f'../solution/{solStringName}.h5', 'r')
u_h5 = file['U'][:, -1]
v_h5 = file['V'][:, -1]
k_h5 = file['TKE'][:, -1]
T_h5 = file['Temp'][:, -1]
z_h5 = file['z'][:]
file.close()

# LES Data Variables
les_variables = {
    "u": {"index": 1, "ylabel": "Height (m)", "xlabel": "x-velocity (u) [m/s]"},
    "v": {"index": 2, "ylabel": "Height (m)", "xlabel": "y-velocity (v) [m/s]"},
    "k": {"index": 5, "ylabel": "Height (m)", "xlabel": "TKE (m²/s²)"},
    "T": {"index": 3, "ylabel": "Height (m)", "xlabel": "Potential Temperature (K)"},
}

# Function to align dimensions
def align_dimensions(array1, array2):
    min_length = min(array1.shape[0], array2.shape[0])
    return array1[:min_length], array2[:min_length]

# Plot LES and .h5 data for the final configuration
for var, details in les_variables.items():
    plt.figure(figsize=(10, 8))
    
    # Plot LES simulation data for the last configuration
    for name in sim_names:
        height = les_data[name]['A_data'][0, :]  # Assuming height is always the first row in A_data
        if var in ['u', 'v', 'T']:
            variable = les_data[name]['A_data'][details["index"], :]
        else:
            variable = les_data[name]['B_data'][details["index"], :]

        # Align dimensions of height and variable
        height, variable = align_dimensions(height, variable)

        plt.plot(variable, height, label=f'LES {name}')
    
    # Plot .h5 file data for the final configuration
    h5_variable = {"u": u_h5, "v": v_h5, "k": k_h5, "T": T_h5}[var]
    z_h5_aligned, h5_variable_aligned = align_dimensions(z_h5, h5_variable)

    plt.plot(h5_variable_aligned, z_h5_aligned, label=f'.h5 {var.upper()} (final)', linestyle='--', linewidth=2, color='black')
    
    # Plot settings
    plt.xlabel(details["xlabel"])
    plt.ylabel(details["ylabel"])
    plt.title(f'{var.upper()} Final State: LES Simulations and .h5 Data')
    plt.legend()
    plt.grid()
    plt.show()

    # save the plots
    plt.savefig(f'les_vs_h5_{var}.png')
    
