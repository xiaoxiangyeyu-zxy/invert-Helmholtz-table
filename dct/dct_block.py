import numpy as np
import cv2

x_shape = 141
y_shape = 101

x_compression = 28
y_compression = 20

block_x_number = 4
block_y_number = 4

x_block = 35
y_block = 25


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
        T_dct[i*x_block:(i+1)*x_block, j*y_block:(j+i)*y_block] = block_img_dct(
            T[i*x_block:(i+1)*x_block, j*y_block:(j+i)*y_block], int(x_compression/block_x_number
                                                                     ), int(y_compression/block_y_number))
        P_dct[i*x_block:(i+1)*x_block, j*y_block:(j+i)*y_block] = block_img_dct(
            P[i*x_block:(i+1)*x_block, j*y_block:(j+i)*y_block], int(x_compression/block_x_number
                                                                     ), int(y_compression/block_y_number))

# T_dct = whole_img_dct(T[:x_shape-1, :y_shape-1], x_compression, y_compression)
# P_dct = whole_img_dct(P[:x_shape-1, :y_shape-1], x_compression, y_compression)
#
# Tm_dct = whole_img_dct(Tm[:x_shape-1, :y_shape-1], x_compression, y_compression)
# Pm_dct = whole_img_dct(Pm[:x_shape-1, :y_shape-1], x_compression, y_compression)

Tm_cut = Tm[:x_shape-1, :y_shape-1]
Pm_cut = Pm[:x_shape-1, :y_shape-1]

# print(np.around(Tm_cut-10**T_dct, 16))
# print(np.around(Tm_cut-Tm_dct, 16))

print(np.mean((abs(Tm_cut-10**T_dct))/Tm_cut))
# print(np.mean((abs(Tm_cut-Tm_dct))/Tm_cut))

# print(np.around(Pm_cut-10**P_dct, 16))
# print(np.around(Pm_cut-Pm_dct, 16))

print(np.mean((abs(Pm_cut-10**P_dct))/Pm_cut))
# print(np.mean((abs(Pm_cut-Pm_dct))/Pm_cut))
