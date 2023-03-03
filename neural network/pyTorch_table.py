import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

interlayer_T1 = 200  # interlayer of network for T
interlayer_P1 = 200  # interlayer of network for P
iteration_T1 = 10000  # iteration times of network for T
iteration_P1 = 10000  # iteration times of network for P


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)  # first interlayer
        self.hidden2 = nn.Linear(n_hidden, n_hidden)  # second interlayer
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, in_put):
        out = self.hidden1(in_put)
        out = torch.sigmoid(out)  # activation function = sigmoid
        out = self.hidden2(out)
        out = torch.sigmoid(out)  # activation function = sigmoid
        out = self.predict(out)

        return out


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

X = X.astype(np.float32)
T = T.astype(np.float32)
P = P.astype(np.float32)

X_torch = Variable(torch.from_numpy(X))
T_torch = Variable(torch.from_numpy(T.reshape(-1, 1)))
P_torch = Variable(torch.from_numpy(P.reshape(-1, 1)))

net_T = Net(2, interlayer_T1, 1)
optimizer = torch.optim.Adam(net_T.parameters(), lr=0.01)  # lr=learning rate
loss_func = torch.nn.MSELoss()  # loss function = mse

loss_now = 10086
for i in range(iteration_T1):
    prediction = net_T(X_torch)  # predict result
    loss = loss_func(prediction, T_torch)  # calculate loss rate

    optimizer.zero_grad()  # set grad to zero
    loss.backward()  # loss back propagation
    optimizer.step()  # grad optimize

    if round(float(loss_now), 4) > round(float(loss), 4):
        print(float(loss), i+1)
        loss_now = loss

net_P = Net(2, interlayer_P1, 1)
optimizer = torch.optim.Adam(net_P.parameters(), lr=0.01)  # lr=learning rate
loss_func = torch.nn.MSELoss()  # loss function = mse

loss_now = 10086
for i in range(iteration_P1):
    prediction = net_P(X_torch)  # predict result
    loss = loss_func(prediction, P_torch)  # calculate loss rate

    optimizer.zero_grad()  # set grad to zero
    loss.backward()  # loss back propagation
    optimizer.step()  # grad optimize

    if round(float(loss_now), 4) > round(float(loss), 4):
        print(float(loss), i+1)
        loss_now = loss


ttrain = net_T(X_torch).detach().numpy().reshape(1, -1)[0]
ptrain = net_P(X_torch).detach().numpy().reshape(1, -1)[0]


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
