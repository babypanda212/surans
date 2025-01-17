import matplotlib.pyplot as plt
import numpy as np

path = "gabls_wind"

u = np.load(f"test_{path}_u.npy")
v = np.load(f"test_{path}_v.npy")
T = np.load(f"test_{path}_T.npy")
k = np.load(f"test_{path}_k.npy")

print(u.shape, v.shape, T.shape, k.shape)

# Create a vertical grid for plotting
z = np.linspace(0, 400, len(u))  # Assuming the height is 400 meters and grid matches u size

# Plot each profile
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(u, z)
plt.xlabel("u (m/s)")
plt.ylabel("Height (m)")
plt.title("Initial Velocity u")

plt.subplot(2, 2, 2)
plt.plot(v, z)
plt.xlabel("v (m/s)")
plt.ylabel("Height (m)")
plt.title("Initial Velocity v")

plt.subplot(2, 2, 3)
plt.plot(T, z)
plt.xlabel("Temperature (K)")
plt.ylabel("Height (m)")
plt.title("Initial Temperature")

plt.subplot(2, 2, 4)
plt.plot(k, z)
plt.xlabel("TKE (m²/s²)")
plt.ylabel("Height (m)")
plt.title("Initial Turbulent Kinetic Energy")

plt.tight_layout()
plt.show()
