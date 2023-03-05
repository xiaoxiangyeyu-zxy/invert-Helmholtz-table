from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.layers import Activation
import plot_fig

epoch_T = 50
batch_Size_T = 32
epoch_P = 50
batch_Size_P = 32

x_shape = 141
y_shape = 101


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

print("lg(press):", P)
print("lg(press_train):", ptrain)
print("press:", Pm)
print("press_train:", ptrainm)

R2P = 1 - np.sum(np.array(P-ptrain)**2)/np.sum(np.array(P-np.mean(P))**2)
R2Pm = 1 - np.sum(np.array(Pm-10**ptrain)**2)/np.sum(np.array(Pm-np.mean(Pm))**2)
print("R2P=", R2P)
print("R2Pm=", R2Pm)

x_plot = 10**x1
y_plot = 10**x2
plot_fig.plot_fig(Pm, Tm, ptrainm, ttrainm, x_plot, y_plot, x_shape, y_shape)

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
