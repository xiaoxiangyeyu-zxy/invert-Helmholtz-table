from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

x1, x2 = np.mgrid[-10:4:141j, 16:26:101j]  # 141*101
v1 = np.ravel(x1)  # 2d->1d
v2 = np.ravel(x2)  # 2d->1d
v1 = 10**v1
v2 = 10**v2
X = np.stack((v1, v2), axis=1)  # reshape to (14241)*2
print(v1)
print(v2)
data = np.loadtxt("table.txt")
T = data[:, 0]
P = data[:, 1]
tt = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', alpha=0.0001,)
pp = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', alpha=0.0001,)
tt.fit(X, T)
pp.fit(X, P)
ttrain = tt.predict(X)
pptrain = pp.predict(X)
print(T)
print(ttrain)
print(P)
print(pptrain)

# # calculate R**2
# Et = np.mean(Y)
# print(Et)
# tminusp = yptrain - Y
# tminusp2 = [x*y for x, y in zip(tminusp, tminusp)]
# tminusEt2 = [(x-Et)**2 for x in Y]
# R2 = 1 - np.sum([x/y for x, y in zip(tminusp2, tminusEt2)])
# print(R2)

# plt.figure()
# Xp1, Yp1 = np.mgrid[-1:1:200j, -1:1:200j]
# Zp1 = Y.reshape(200, 200)
# plt.contourf(Xp1, Yp1, Zp1, np.linspace(0., 2., 11))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('true')
# plt.savefig('true.pdf')
#
# plt.figure()
# Xp2, Yp2 = np.mgrid[-1:1:200j, -1:1:200j]
# Zp2 = yptrain.reshape(200, 200)
# plt.contourf(Xp2, Yp2, Zp2, np.linspace(0., 2., 11))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('train')
# plt.savefig('train.pdf')
