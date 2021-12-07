"""
功能测试代码
"""
import os
import random

from numpy import shape

import utils
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# path = "/data/zengnanrong/LUNG_SEG/E0001001V1/E0001001V1FC03/1.2.392.200036.9116.2.5.1.48.1215508268.1254185734.309395.dcm"
#
# image = sitk.ReadImage(path)
# image_array = np.squeeze(sitk.GetArrayFromImage(image))
# plt.imshow(image_array)
# plt.show()
#
# lungmask_path = path.replace('CTDATA', 'R231')
# lungmask_image = sitk.ReadImage(lungmask_path)
# lungmask_image_array = np.squeeze(sitk.GetArrayFromImage(lungmask_image))
#
# plt.imshow(lungmask_image_array)
# plt.show()
#
# height = image_array.shape[0]
# width = image_array.shape[1]
#
# for h in range(height):
#     for w in range(width):
#         if lungmask_image_array[h][w] == 0:
#             # 将非肺区域置0
#             image_array[h][w] = 0
#
# plt.imshow(image_array)
# plt.show()

# list = []
# list.append({'key_1': 1, 'key_2': 'a'})
# list.append({'key_1': 2, 'key_2': 'b'})
# list.append({'key_1': 3, 'key_2': 'c'})
# list.append({'key_1': 4, 'key_2': 'd'})
# list.append({'key_1': 5, 'key_2': 'e'})
# list.append({'key_1': 6, 'key_2': 'f'})
#
# print(list)
#
# random.shuffle(list)
#
# print(list)


# local_filename = "/data/zengnanrong/CTDATA/E0001001V1/E0001001V1FC03/1.2.392.200036.9116.2.5.1.48.1215508268.1254185772.683629.dcm"
# # filename = os.path.split(local_filename)[1]
# # filedir = os.path.split(local_filename)[0]
# # print(filedir)
# # print(filename)
# input_root_path = "/data/zengnanrong/CTDATA/"
# output_root_path = "/data/zengnanrong/R231/"
# output = local_filename.replace(input_root_path, output_root_path)
# print(output)
#
# from pathos.multiprocessing import ProcessingPool as Pool
#
#
# def test1(x, y, z):
#     print("x:%d, y:%s, z =%s" % (x, y, z))
#
#
# def test2(x):
#     print(x)
#
#
# x = [1, 2, 3, 4]
# y = ['y', 'y', 'y', 'y']
# z = ['y', 'y', 'y', 'y']
# pool = Pool()
# pool.map(test2, x)
# pool.close()
# pool.join()
