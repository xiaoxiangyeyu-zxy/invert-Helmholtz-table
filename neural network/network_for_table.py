from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors

x1, x2 = np.mgrid[-10:4:141j, 16:26:101j]  # 141*101
v1 = np.ravel(x1)  # lg(density) 2d->1d
v2 = np.ravel(x2)  # lg(energy) 2d->1d
v1m = 10**v1  # density
v2m = 10**v2  # energy
X = np.stack((v1, v2), axis=1)  # reshape to (14241)*2
# print(v1)
# print(v2)
data = np.loadtxt("table.txt")
Tm = data[:, 0]  # temperature
Pm = data[:, 1]  # intensity of pressure
T = np.log10(Tm)  # lg(tem)
P = np.log10(Pm)  # lg(press)
tt = MLPRegressor(hidden_layer_sizes=(200, 50, 25), activation='relu', solver='adam', alpha=0.0001,)
pp = MLPRegressor(hidden_layer_sizes=(200, 50, 25), activation='relu', solver='adam', alpha=0.0001,)
tt.fit(X, T)
pp.fit(X, P)
ttrain = tt.predict(X)
ptrain = pp.predict(X)
ttrainm = 10**ttrain
ptrainm = 10**ptrain

print("lg(temp):", T)
print("lg(temp_train):", ttrain)
print("temp:", Tm)
print("temp_train", ttrainm)

R2T = 1 - np.sum(np.array(T-ttrain)**2)/np.sum(np.array(T-np.mean(T))**2)
R2Tm = 1 - np.sum(np.array(Tm-10**ttrain)**2)/np.sum(np.array(Tm-np.mean(Tm))**2)
print("R2T=", R2T)
print("R2Tm=", R2Tm)

lev_exp = np.linspace(np.floor(np.log10(Tm.min())), np.ceil(np.log10(Tm.max())), 40)
levs = np.power(10, lev_exp)

plt.figure()
Xp1, Yp1 = 10**(np.mgrid[-10:4:141j, 16:26:101j])
Zp1 = Tm.reshape(141, 101)
# plt.contourf(Xp1, Yp1, Zp1, np.linspace(0.e10, 9.0e10, 10))
plt.contourf(Xp1, Yp1, Zp1, levs, norm=colors.LogNorm())
plt.xlabel('density')
plt.ylabel('energy')
plt.xscale('log')
plt.yscale('log')
plt.title('true_T')
plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
plt.savefig('true_T.pdf')

plt.figure()
Xp2, Yp2 = 10**(np.mgrid[-10:4:141j, 16:26:101j])
Zp2 = ttrainm.reshape(141, 101)
# plt.contourf(Xp2, Yp2, Zp2, np.linspace(0.e10, 9.0e10, 10))
plt.contourf(Xp2, Yp2, Zp2, levs, norm=colors.LogNorm())
plt.xlabel('density')
plt.ylabel('energy')
plt.xscale('log')
plt.yscale('log')
plt.title('train_T')
plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
plt.savefig('train_T.pdf')

print("lg(press):", P)
print("lg(press_train):", ptrain)
print("press:", Pm)
print("press_train:", ptrainm)

R2P = 1 - np.sum(np.array(P-ptrain)**2)/np.sum(np.array(P-np.mean(P))**2)
R2Pm = 1 - np.sum(np.array(Pm-10**ptrain)**2)/np.sum(np.array(Pm-np.mean(Pm))**2)
print("R2P=", R2P)
print("R2Pm=", R2Pm)

lev_exp = np.linspace(np.floor(np.log10(Pm.min())), np.ceil(np.log10(Pm.max())), 40)
levs = np.power(10, lev_exp)

plt.figure()
Xp3, Yp3 = 10**(np.mgrid[-10:4:141j, 16:26:101j])
Zp3 = Pm.reshape(141, 101)
# plt.contourf(Xp3, Yp3, Zp3, np.linspace(0.e29, 3.5e29, 8))
plt.contourf(Xp3, Yp3, Zp3, levs, norm=colors.LogNorm())
plt.xlabel('density')
plt.ylabel('energy')
plt.xscale('log')
plt.yscale('log')
plt.title('true_P')
plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
plt.savefig('true_P.pdf')

plt.figure()
Xp4, Yp4 = 10**(np.mgrid[-10:4:141j, 16:26:101j])
Zp4 = ptrainm.reshape(141, 101)
# plt.contourf(Xp4, Yp4, Zp4, np.linspace(0.e29, 3.5e29, 8))
plt.contourf(Xp4, Yp4, Zp4, levs, norm=colors.LogNorm())
plt.xlabel('density')
plt.ylabel('energy')
plt.xscale('log')
plt.yscale('log')
plt.title('train_P')
plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
plt.savefig('train_P.pdf')
