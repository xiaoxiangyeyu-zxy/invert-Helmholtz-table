import numpy as np

# read the data from the file "table.txt"
data = np.loadtxt("table.txt")
# print(data)

# the first column of data is temperature, the second column of data is intensity of pressure
T = data[:, 0]
P = data[:, 1]

lgdenmax = 4  # lg(density upper limit of net electron)
indexden = int((lgdenmax + 10) * 10 + 1)  # density total index

lgEmin = 16.  # lg(energy upper limit)
lgEmax = 26.  # lg(energy lower limit)
dE = 0.1      # interval of lgE
indexE = int((lgEmax-lgEmin)/dE)+1  # energy total index

# the first index is density, the second index is energy
# density: lgden from -10 to 4, interval = 0.1, total points are 141
# energy: lgE from 16 to 26, interval = 0.1, total points are 101
Tuse = T.reshape(indexden, indexE)
Puse = P.reshape(indexden, indexE)
