import numpy as np

# read the data from the file "Helmholtz_29_21_den_energy.txt"
with open("visual.txt", "r") as f:
    line1 = f.readline()
    l1 = line1.split()
    line2 = f.readline()
    l2 = line2.split()
    data = np.loadtxt(f)
f.close()

den_index = int(l1[1])
dDen = float(l1[2])
lgdenmin = math.log10(float(l1[3]))
energy_index = int(l2[1])
dE = float(l2[2])
lgenergymin = math.log10(float(l2[3]))

# print(data)

# the first column of data is temperature, the second column of data is intensity of pressure
T = data[:, 0]
P = data[:, 1]

# the first index is density, the second index is energy
Tuse = T.reshape(den_index, energy_index)
Puse = P.reshape(den_index, energy_index)
