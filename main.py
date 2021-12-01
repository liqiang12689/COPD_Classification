import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

path = "/data/zengnanrong/CTDATA_test/E0001001V1/E0001001V1FC03/1.2.392.200036.9116.2.5.1.48.1215508268.1254185544.78233.dcm"

image = sitk.ReadImage(path)
image_array = np.squeeze(sitk.GetArrayFromImage(image))
plt.imshow(image_array)
plt.show()

print(image_array.shape)
