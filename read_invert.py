import numpy as np

# read the data from the file "Helmholtz_29_21_den_energy.txt"
with open("Helmholtz_29_21_den_energy.txt", "r") as f:
    den_index = f.readline()
    energy_index = f.readline()
    lgdenmin = f.readline()
    lgdenmax = f.readline()
    lgenergymin = f.readline()
    lgenergymax = f.readline()
    data = np.loadtxt(f)
f.close()
# print(data)

den_index = int(den_index)
energy_index = int(energy_index)
lgdenmin = float(lgdenmin)
lgdenmax = float(lgdenmax)
lgenergymin = float(lgenergymin)
lgenergymax = float(lgenergymax)

# the first column of data is temperature, the second column of data is intensity of pressure
T = data[:, 0]
P = data[:, 1]

# the first index is density, the second index is energy
Tuse = T.reshape(den_index, energy_index)
Puse = P.reshape(den_index, energy_index)
