import numpy as np
import plot_fig

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
    minusde = dE2 - de
    minusdden = dden2 - dden
    t_inter = abs(de * dden) / (dE2 * dden2) * T2[ii+1, jj+1] + abs(minusde * dden) / (dE2 * dden2) * T2[ii+1, jj] + \
        abs(de * minusdden) / (dE2 * dden2) * T2[ii, jj+1] + abs(minusde * minusdden) / (dE2 * dden2) * T2[
                  ii, jj]
    p_inter = abs(de * dden) / (dE2 * dden2) * P2[ii+1, jj+1] + abs(minusde * dden) / (dE2 * dden2) * P2[ii+1, jj] + \
        abs(de * minusdden) / (dE2 * dden2) * P2[ii, jj+1] + abs(minusde * minusdden) / (dE2 * dden2) * P2[
                  ii, jj]
    return t_inter, p_inter


T_result = [[] for i in range(x_shape1 - 1)]
P_result = [[] for j in range(x_shape1 - 1)]
for i in range(x_shape1 - 1):
    lgden = lgdenmin + i * dden1
    indexden = int((lgden - lgdenmin) / dden2)
    delta_den = (lgden - lgdenmin) % dden2
    for j in range(y_shape1 - 1):
        lgE = lgEmin + j * dE1
        indexE = int((lgE - lgEmin) / dE2)
        delta_E = (lgE - lgEmin) % dE2
        T_use, P_use = interpolate(indexden, indexE, delta_E, delta_den)
        T_result[i].append(T_use)
        P_result[i].append(P_use)

T_result = np.array(T_result)
T_cut = T1[:x_shape1-1, :y_shape1-1]
Tm_cut = Tm1[:x_shape1-1, :y_shape1-1]

P_result = np.array(P_result)
P_cut = P1[:x_shape1-1, :y_shape1-1]
Pm_cut = Pm1[:x_shape1-1, :y_shape1-1]

print(np.mean(abs(Tm_cut-10**T_result)/Tm_cut))
print(T_result)
print(T_cut)

print(np.mean(abs(Pm_cut-10**P_result)/Pm_cut))
print(P_result)
print(P_cut)

x1, x2 = np.mgrid[-10:3.9:140j, 16:25.9:100j]
x1 = 10**x1
x2 = 10**x2
plot_fig.plot_fig(Tm_cut, 10**T_result, x1, x2, x_shape1-1, y_shape1-1, "T")
plot_fig.plot_fig(Pm_cut, 10**P_result, x1, x2, x_shape1-1, y_shape1-1, "P")