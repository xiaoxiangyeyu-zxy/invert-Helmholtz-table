from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

x1, x2 = np.mgrid[-1:1:200j, -1:1:200j]  # 200*200
v1 = np.ravel(x1)  # 2d->1d
v2 = np.ravel(x2)  # 2d->1d
Y = v1**2 + v2**2  # get x1**2+x2**2
X = np.stack((v1, v2), axis=1)  # reshape to 40000*2
nn = MLPRegressor(hidden_layer_sizes=(500, 200), activation='relu', solver='adam', alpha=0.0001,)
nn.fit(X, Y)
yptrain = nn.predict(X)

# zzz = nn.predict(np.array([[0., 0.]]))
# print(yptrain)
# print(Y)
# print(zzz)

# calculate R**2
Et = np.mean(Y)
print(Et)
tminusp = yptrain - Y
tminusp2 = [x*y for x, y in zip(tminusp, tminusp)]
tminusEt2 = [(x-Et)**2 for x in Y]
R2 = 1 - np.sum([x/y for x, y in zip(tminusp2, tminusEt2)])
print(R2)

plt.figure()
Xp1, Yp1 = np.mgrid[-1:1:200j, -1:1:200j]
Zp1 = Y.reshape(200, 200)
plt.contourf(Xp1, Yp1, Zp1, np.linspace(0., 2., 11))
plt.xlabel('x')
plt.ylabel('y')
plt.title('true')
plt.savefig('true.pdf')

plt.figure()
Xp2, Yp2 = np.mgrid[-1:1:200j, -1:1:200j]
Zp2 = yptrain.reshape(200, 200)
plt.contourf(Xp2, Yp2, Zp2, np.linspace(0., 2., 11))
plt.xlabel('x')
plt.ylabel('y')
plt.title('train')
plt.savefig('train.pdf')
