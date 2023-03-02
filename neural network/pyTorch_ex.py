import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, in_put):
        out = self.hidden1(in_put)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.predict(out)

        return out


net = Net(1, 20, 1)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(3)+0.1*torch.randn(x.size())
x, y = (Variable(x), Variable(y))


for i in range(5000):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()  # set grad to zero
    loss.backward()
    optimizer.step()

    print(float(loss), i+1)

# print(y)
# print(net(x))
