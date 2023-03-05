from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Activation

epoch_T = 50
batch_Size_T = 32
epoch_P = 50
batch_Size_P = 32

def getmodel():
    model = Sequential()
    model.add(Dense(1, use_bias=False, input_shape=(2,)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


x1, x2 = np.mgrid[-10:4:141j, 16:26:101j]  # 141*101
v1 = np.ravel(x1)  # lg(density) 2d->1d
v2 = np.ravel(x2)  # lg(energy) 2d->1d
v1m = 10**v1  # density
v2m = 10**v2  # energy
X = np.stack((v1, v2), axis=1)  # reshape to (14241)*2
data = np.loadtxt("table.txt")
Tm = data[:, 0]  # temperature
Pm = data[:, 1]  # intensity of pressure
T = np.log10(Tm)  # lg(tem)
P = np.log10(Pm)  # lg(press)

modelt = getmodel()
modelt.fit(X, T, epochs=epoch_T, batch_size=batch_Size_T)
ttrain = modelt.predict(X)

modelp = getmodel()
modelp.fit(X, P, epochs=epoch_P, batch_size=batch_Size_P)
ptrain = modelp.predict(X)

ttrain = ttrain.reshape(1, -1)[0]
ptrain = ptrain.reshape(1, -1)[0]
ttrainm = 10**ttrain
ptrainm = 10**ptrain

print("lg(temp):", T)
print("lg(temp_train):", ttrain)
print("temp:", Tm)
print("temp_train", ttrainm)

R2T = 1 - np.sum(np.array(T-ttrain)**2)/np.sum(np.array(T-np.mean(T))**2)
R2Tm = 1 - np.sum(np.array(Tm-ttrainm)**2)/np.sum(np.array(Tm-np.mean(Tm))**2)
print("R2T=", R2T)
print("R2Tm=", R2Tm)

plt.figure()
Xp1, Yp1 = 10**(np.mgrid[-10:4:141j, 16:26:101j])
Zp1 = Tm.reshape(141, 101)
plt.contourf(Xp1, Yp1, Zp1, np.linspace(0.e10, 9.0e10, 10))
plt.xlabel('density')
plt.ylabel('energy')
plt.title('true_T')
plt.colorbar()
plt.savefig('true_T.png')

plt.figure()
Xp2, Yp2 = 10**(np.mgrid[-10:4:141j, 16:26:101j])
Zp2 = ttrainm.reshape(141, 101)
plt.contourf(Xp2, Yp2, Zp2, np.linspace(0.e10, 9.0e10, 10))
plt.xlabel('density')
plt.ylabel('energy')
plt.title('train_T')
plt.colorbar()
plt.savefig('train_T.png')

print("lg(press):", P)
print("lg(press_train):", ptrain)
print("press:", Pm)
print("press_train:", ptrainm)

R2P = 1 - np.sum(np.array(P-ptrain)**2)/np.sum(np.array(P-np.mean(P))**2)
R2Pm = 1 - np.sum(np.array(Pm-10**ptrain)**2)/np.sum(np.array(Pm-np.mean(Pm))**2)
print("R2P=", R2P)
print("R2Pm=", R2Pm)

plt.figure()
Xp3, Yp3 = 10**(np.mgrid[-10:4:141j, 16:26:101j])
Zp3 = Pm.reshape(141, 101)
plt.contourf(Xp3, Yp3, Zp3, np.linspace(0.e29, 3.5e29, 8))
plt.xlabel('density')
plt.ylabel('energy')
plt.title('true_P')
plt.colorbar()
plt.savefig('true_P.png')

plt.figure()
Xp4, Yp4 = 10**(np.mgrid[-10:4:141j, 16:26:101j])
Zp4 = ptrainm.reshape(141, 101)
plt.contourf(Xp4, Yp4, Zp4, np.linspace(0.e29, 3.5e29, 8))
plt.xlabel('density')
plt.ylabel('energy')
plt.title('train_P')
plt.colorbar()
plt.savefig('train_P.png')

with open("R2.txt", "w") as f:
    f.write("R2T:")
    f.write(str(R2T))
    f.write("\n")
    f.write("R2Tm:")
    f.write(str(R2Tm))
    f.write("\n")
    f.write("R2P:")
    f.write(str(R2P))
    f.write("\n")
    f.write("R2Pm:")
    f.write(str(R2Pm))
