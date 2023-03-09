import numpy as np
import cv2
# import plot_fig

x_shape = 141
y_shape = 101

x_compression = 28
y_compression = 20

# 整张图 DCT 变换
def whole_img_dct(img_f32, x, y):
    img_dct = cv2.dct(img_f32)            # 进行离散余弦变换
    # img_dct_log = np.log(abs(img_dct))    # 进行log处理
    # img_recor = cv2.idct(img_dct)          # 进行离散余弦反变换
    recor_temp = img_dct[0:x_compression, 0:y_compression]
    recor_temp2 = np.zeros(img_f32.shape)
    recor_temp2[0:x_compression, 0:y_compression] = recor_temp
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

T_dct = whole_img_dct(T[:x_shape-1, :y_shape-1], x_compression, y_compression)
# print(T)
# print(T_dct)

Tm_dct = whole_img_dct(Tm[:x_shape-1, :y_shape-1], x_compression, y_compression)
# print(Tm)
# print(Tm_dct)

Tm_cut = Tm[:x_shape-1, :y_shape-1]

print(np.around(Tm_cut-10**T_dct, 16))
print(np.around(Tm_cut-Tm_dct, 16))

print(np.mean((abs(Tm_cut-10**T_dct))/Tm_cut))
print(np.mean((abs(Tm_cut-Tm_dct))/Tm_cut))
