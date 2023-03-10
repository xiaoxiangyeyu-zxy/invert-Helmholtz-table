import numpy as np

lgEmin = 16.
lgEmax = 26.
lgdenmin = -10.
lgdenmax = 4.

dE1 = 0.1
dden1 = 0.1
dE2 = 0.5
dden2 = 0.5

x_shape1 = 141
y_shape1 = 101
x_shape2 = 29
y_shape2 = 21

data1 = np.loadtxt("table.txt")
Tm1 = data1[:, 0]  # temperature
Pm1 = data1[:, 1]  # intensity of pressure
T1 = np.log10(Tm1)  # lg(tem)
P1 = np.log10(Pm1)  # lg(press)
Tm1 = Tm1.reshape(x_shape1, y_shape1)
Pm1 = Pm1.reshape(x_shape1, y_shape1)
T1 = T1.reshape(x_shape1, y_shape1)
P1 = P1.reshape(x_shape1, y_shape1)

data2 = np.loadtxt("table_29_21.txt")
Tm2 = data2[:, 0]  # temperature
Pm2 = data2[:, 1]  # intensity of pressure
T2 = np.log10(Tm2)  # lg(tem)
P2 = np.log10(Pm2)  # lg(press)
Tm2 = Tm2.reshape(x_shape2, y_shape2)
Pm2 = Pm2.reshape(x_shape2, y_shape2)
T2 = T2.reshape(x_shape2, y_shape2)
P2 = P2.reshape(x_shape2, y_shape2)


def interpolate(ii, jj, de, dden):
    d10den = 10**((ii+1)*dden2+lgdenmin) - 10**(ii*dden2+lgdenmin)
    d10E = 10**((jj+1)*dE2+lgEmin) - 10**(jj*dE2+lgEmin)
    minusde = d10E - de
    minusdden = d10den - dden
    t_inter = abs(de * dden) / (d10E * d10den) * Tm2[ii+1, jj+1] + abs(minusde * dden) / (d10E * d10den) * Tm2[ii+1, jj] + \
        abs(de * minusdden) / (d10E * d10den) * Tm2[ii, jj+1] + abs(minusde * minusdden) / (d10E * d10den) * Tm2[
                  ii, jj]
    p_inter = abs(de * dden) / (d10E * d10den) * Pm2[ii+1, jj+1] + abs(minusde * dden) / (d10E * d10den) * Pm2[ii+1, jj] + \
        abs(de * minusdden) / (d10E * d10den) * Pm2[ii, jj+1] + abs(minusde * minusdden) / (d10E * d10den) * Pm2[
                  ii, jj]
    return t_inter, p_inter


Tm_result = [[] for i in range(x_shape1 - 1)]
Pm_result = [[] for j in range(x_shape1 - 1)]
for i in range(x_shape1 - 1):
    lgden = lgdenmin + i * dden1
    indexden = int((lgden - lgdenmin) / dden2)
    delta_den = 10**lgden - 10**(indexden*dden2+lgdenmin)
    for j in range(y_shape1 - 1):
        lgE = lgEmin + j * dE1
        indexE = int((lgE - lgEmin) / dE2)
        delta_E = 10**lgE - 10**(indexE*dE2+lgEmin)
        T_use, P_use = interpolate(indexden, indexE, delta_E, delta_den)
        Tm_result[i].append(T_use)
        Pm_result[i].append(P_use)

Tm_result = np.array(Tm_result)
T_cut = T1[:x_shape1-1, :y_shape1-1]
Tm_cut = Tm1[:x_shape1-1, :y_shape1-1]

Pm_result = np.array(Pm_result)
P_cut = P1[:x_shape1-1, :y_shape1-1]
Pm_cut = Pm1[:x_shape1-1, :y_shape1-1]

print(np.mean(abs(Tm_cut-Tm_result)/Tm_cut))
print(Tm_result)
print(Tm_cut)

print(np.mean(abs(Pm_cut-Pm_result)/Pm_cut))
print(Pm_result)
print(Pm_cut)
