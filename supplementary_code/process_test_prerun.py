import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter


plt.close('all')

solStringName = 'ini_wind'

file = h5py.File('../solution/' + solStringName + '.h5' , 'r+')
u    = file['U'][:]
v    = file['V'][:]
k    = file['TKE'][:]
T    = file['Temp'][:]
z    = file['z'][:] # meters
dt   = file['dt'][:] # seconds
SimStart = int(file['SimStart' ][:])
SimEnd = int(file['SimEnd' ][:])
file.close()


# Define variables to plot
variables_to_plot = ['u']  # Choose any combination: ['u', 'v', 'k', 'T']

data_map = {
    'u': u,
    'v': v,
    'k': k,
    'T': T,
}

# create grid for plotting
nt = SimEnd - SimStart
t    = np.linspace(0, nt * dt / (3600), nt+1) # in hours
X, Y = np.meshgrid(t, z)
# Iterate over variables_to_plot and generate plots
for var in variables_to_plot:
    plt.figure()
    plt.pcolor(X, Y, data_map[var])
    plt.xlabel('time [h]')
    plt.ylabel('z [m]')
    plt.colorbar()
    plt.title(var.upper())
    plt.show()

# Animate the evolution of the wind
fig, ax = plt.subplots()
plt.xlabel('u [m/s]')
plt.ylabel('z [m]')
line, = ax.plot(u[:, 0], z)
ani = animation.FuncAnimation(fig, lambda i: line.set_xdata(u[:, i]), frames=179, interval=1, blit=True, repeat=False)
writer = FFMpegWriter(fps=60)
ani.save('u.mp4', writer=writer)


# plot the final state of the wind
plt.figure()
plt.plot(u[:, -1], z)
plt.xlabel('u [m/s]')
plt.ylabel('z [m]')
plt.title('Final state of the wind')

# save the plot
plt.savefig('final_wind.png')
plt.show()
