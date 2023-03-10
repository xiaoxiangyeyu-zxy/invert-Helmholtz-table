import numpy as np
import cv2
import plot_fig

x_shape = 141
y_shape = 101

x_compression = 30
y_compression = 20

block_x_number = 10
block_y_number = 10

x_block = int((x_shape-1)/block_x_number)
y_block = int((y_shape-1)/block_y_number)

x_in = int(x_compression / block_x_number)
y_in = int(y_compression / block_y_number)

def block_img_dct(img_f32, x, y):
    img_dct = cv2.dct(img_f32)
    # img_dct_log = np.log(abs(img_dct))
    # img_recor = cv2.idct(img_dct)
    recor_temp = img_dct[0:x, 0:y]
    recor_temp2 = np.zeros(img_f32.shape)
    recor_temp2[0:x, 0:y] = recor_temp
    img_recor1 = cv2.idct(recor_temp2)
    return img_recor1


data = np.loadtxt("table.txt")
Tm = data[:, 0]  # temperature
Pm = data[:, 1]  # intensity of pressure
T = np.log10(Tm)  # lg(tem)
P = np.log10(Pm)  # lg(press)

T = T.reshape(141, 101)
P = P.reshape(141, 101)
Tm = Tm.reshape(141, 101)
Pm = Pm.reshape(141, 101)

T_dct = np.zeros((x_shape-1, y_shape-1))
P_dct = np.zeros((x_shape-1, y_shape-1))
for i in range(block_x_number):
    for j in range(block_y_number):
        T_use = block_img_dct(T[i*x_block:(i+1)*x_block, j*y_block:(j+1)*y_block], x_in, y_in)
        T_dct[i * x_block:(i + 1) * x_block, j * y_block:(j + 1) * y_block] = T_use
        P_use = block_img_dct(P[i*x_block:(i+1)*x_block, j*y_block:(j+1)*y_block], x_in, y_in)
        P_dct[i * x_block:(i + 1) * x_block, j * y_block:(j + 1) * y_block] = P_use

Tm_dct = np.zeros((x_shape-1, y_shape-1))
Pm_dct = np.zeros((x_shape-1, y_shape-1))
for i in range(block_x_number):
    for j in range(block_y_number):
        Tm_use = block_img_dct(Tm[i*x_block:(i+1)*x_block, j*y_block:(j+1)*y_block], x_in, y_in)
        Tm_dct[i * x_block:(i + 1) * x_block, j * y_block:(j + 1) * y_block] = Tm_use
        Pm_use = block_img_dct(Pm[i*x_block:(i+1)*x_block, j*y_block:(j+1)*y_block], x_in, y_in)
        Pm_dct[i * x_block:(i + 1) * x_block, j * y_block:(j + 1) * y_block] = Pm_use

Tm_cut = Tm[:x_shape-1, :y_shape-1]
Pm_cut = Pm[:x_shape-1, :y_shape-1]

# print(np.around(Tm_cut-10**T_dct, 16))
# print(np.around(Tm_cut-Tm_dct, 16))

print(np.mean((abs(Tm_cut-10**T_dct))/Tm_cut))
print(np.mean((abs(Tm_cut-Tm_dct))/Tm_cut))

# print(np.around(Pm_cut-10**P_dct, 16))
# print(np.around(Pm_cut-Pm_dct, 16))

print(np.mean((abs(Pm_cut-10**P_dct))/Pm_cut))
print(np.mean((abs(Pm_cut-Pm_dct))/Pm_cut))

x1, x2 = np.mgrid[-10:3.9:140j, 16:25.9:100j]
x1 = 10**x1
x2 = 10**x2
plot_fig.plot_fig(Tm_cut, 10**T_dct, x1, x2, x_shape-1, y_shape-1, "T")
plot_fig.plot_fig(Pm_cut, 10**P_dct, x1, x2, x_shape-1, y_shape-1, "P")
