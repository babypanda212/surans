import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as animation


plt.close('all')

solStringName = 'test_run'

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

# create grid for plotting
nt = SimEnd - SimStart
t    = np.linspace(0, nt * dt / (3600), nt+1) # in hours
X, Y = np.meshgrid(t, z)


plt.pcolor(X, Y, k)
plt.xlabel('time [h]')
plt.ylabel('z [h]')
plt.colorbar()
plt.title('TKE')

fig, ax = plt.subplots()

ax.set_xlabel('u[m/s]')
ax.set_ylabel('z[m]')
line, = ax.plot(u[:, 0], z)

def animate(i):
    print(i)
    line.set_xdata(u[:, i])  # update the data.
    return line,

ani = animation.FuncAnimation(fig, animate, frames=179, interval=1, blit=True, repeat=False)

ani.save('u_clean.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

plt.show()
