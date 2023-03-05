import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors


def plot_fig(pm, tm, ptrainm, ttrainm, x1, x2, x_shape, y_shape):
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
    plt.title('true_t')
    plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
    plt.savefig('true_t.png')

    plt.figure()
    xp2, yp2 = x1, x2
    zp2 = ttrainm.reshape(x_shape, y_shape)
    plt.contourf(xp2, yp2, zp2, levs, norm=colors.LogNorm())
    plt.xlabel('density')
    plt.ylabel('energy')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('train_t')
    plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
    plt.savefig('train_t.png')

    lev_exp = np.linspace(np.floor(np.log10(pm.min())), np.ceil(np.log10(pm.max())), 40)
    levs = np.power(10, lev_exp)

    plt.figure()
    xp3, yp3 = x1, x2
    zp3 = pm.reshape(x_shape, y_shape)
    plt.contourf(xp3, yp3, zp3, levs, norm=colors.LogNorm())
    plt.xlabel('density')
    plt.ylabel('energy')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('true_p')
    plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
    plt.savefig('true_p.png')

    plt.figure()
    xp4, yp4 = x1, x2
    zp4 = ptrainm.reshape(x_shape, y_shape)
    plt.contourf(xp4, yp4, zp4, levs, norm=colors.LogNorm())
    plt.xlabel('density')
    plt.ylabel('energy')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('train_p')
    plt.colorbar(format=ticker.LogFormatter(10, labelOnlyBase=False))
    plt.savefig('train_p.png')
