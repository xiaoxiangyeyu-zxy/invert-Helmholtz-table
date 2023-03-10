import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors


def plot_fig(tm, ttrainm, x1, x2, x_shape, y_shape, name):
    lev_exp = np.linspace(np.floor(np.log10(tm.min())), np.ceil(np.log10(tm.max())), 40)
    levs = np.power(10, lev_exp)

    plt.figure()
    xp1, yp1 = x1, x2
    zp1 = tm.reshape(x_shape, y_shape)
    plt.contourf(xp1, yp1, zp1, levs, norm=colors.LogNorm())
    plt.xlabel('density')
    plt.ylabel('energy')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('true_'+name)
    plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
    plt.savefig('true_'+name+'.png')

    plt.figure()
    xp2, yp2 = x1, x2
    zp2 = ttrainm.reshape(x_shape, y_shape)
    plt.contourf(xp2, yp2, zp2, levs, norm=colors.LogNorm())
    plt.xlabel('density')
    plt.ylabel('energy')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('train_'+name)
    plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
    plt.savefig('train_'+name+'.png')
