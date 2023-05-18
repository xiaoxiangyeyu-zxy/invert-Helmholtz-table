import numpy as np
import math

def den_ene_to_p(den, ene):
    with open("visual.txt", "r") as yu:
        line1 = yu.readline()
        l1 = line1.split()
        line2 = yu.readline()
        l2 = line2.split()
        data = np.loadtxt(yu)
    yu.close()

    den_index = int(l1[1])
    dDen = float(l1[2])
    lgdenmin = math.log10(float(l1[3]))
    energy_index = int(l2[1])
    dE = float(l2[2])
    lgenergymin = math.log10(float(l2[3]))

    Tm = data[:, 0]  # temperature
    Pm = data[:, 1]  # intensity of pressure
    lgT = np.log10(Tm)  # lg(tem)
    lgP = np.log10(Pm)  # lg(press)
    lgT = lgT.reshape(den_index, energy_index)
    lgP = lgP.reshape(den_index, energy_index)


    def interpolate_yu(ii, jj, de, dden):
        minusde = dE - de
        minusdden = dDen - dden
        t_inter = abs(de * dden) / (dE * dDen) * lgT[ii+1, jj+1] + abs(minusde * dden) / (dE * dDen) * lgT[ii+1, jj] + \
            abs(de * minusdden) / (dE * dDen) * lgT[ii, jj+1] + abs(minusde * minusdden) / (dE * dDen) * lgT[
                      ii, jj]
        p_inter = abs(de * dden) / (dE * dDen) * lgP[ii+1, jj+1] + abs(minusde * dden) / (dE * dDen) * lgP[ii+1, jj] + \
            abs(de * minusdden) / (dE * dDen) * lgP[ii, jj+1] + abs(minusde * minusdden) / (dE * dDen) * lgP[
                      ii, jj]
        return t_inter, p_inter

    dimension_1 = den.shape[0]
    dimension_2 = den.shape[1]
    dimension_3 = den.shape[2]

    T_out = [[[]for j in range(dimension_2)]for i in range(dimension_1)]
    P_out = [[[] for j in range(dimension_2)] for i in range(dimension_1)]

    for i in range(dimension_1):
        for j in range(dimension_2):
            for k in range(dimension_3):
                lgden = np.log10(den[i][j][k])
                indexden = int((lgden - lgdenmin) / dDen)
                delta_den = lgden - lgdenmin - dDen*indexden

                lgE = np.log10(ene[i][j][k])
                indexE = int((lgE - lgenergymin) / dE)
                delta_E = lgE - lgenergymin - dE*indexE
                # print(lgden,delta_den)
                T_use, P_use = interpolate_yu(indexden, indexE, delta_E, delta_den)
                T_out[i][j].append(10**T_use)
                P_out[i][j].append(10**P_use)
    return np.array(P_out)


# test_den = [[[pow(10, 2.3)], [pow(10, 2.4)]], [[pow(10, 2.3)], [pow(10, 2.3)]]]
# test_ene = [[[pow(10, 20.4)], [pow(10, 20.4)]], [[pow(10, 20.4)], [pow(10, 20.4)]]]
# print(den_ene_to_p(np.array(test_den), np.array(test_ene)))
