from keras.models import Sequential
from keras.layers import Dense
import numpy as np

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
modelt.fit(X, T, epochs=40, batch_size=32)
ttrain = modelt.predict(X)
print(T)
print(ttrain)
print(ttrain.shape)


modelp = getmodel()
modelp.fit(X, P, epochs=40, batch_size=32)
ptrain = modelp.predict(X)
print(P)
print(ptrain)
