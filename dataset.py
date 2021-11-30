import os

import utils

import pandas as pd
import SimpleITK as sitk
import numpy as np


def label_preprocess(label_path, output_path):
    """
    将label文件表中的文件名与CTDATA文件夹中的名称对应
    表里的V2--->CT图像的V1
    表里的V3--->CT图像的V2
    表里的V4--->CT图像的V3
    :param label_path:
    :param output_path:
    :return:
    """

    label_data = pd.read_excel(os.path.join(label_path), sheet_name='Sheet1')
    for i in range(len(label_data['subject'])):
        version_num = int(label_data['subject'][i][9]) - 1
        label_data['subject'][i] = label_data['subject'][i][:9] + str(version_num)

    pd.DataFrame(label_data).to_excel(os.path.join(output_path), sheet_name='Sheet1')


def load_data_label(data_root_path, label_path):
    """
    加载每一张DICOM图像，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :return:
    """
    ct_dir = []
    for item in os.listdir(data_root_path):
        if os.path.isdir(os.path.join(data_root_path, item)):
            ct_dir.append(item)

    # 确保与label文件中的名称顺序对应
    ct_dir.sort()

    label_list = pd.read_excel(os.path.join(label_path), sheet_name='Sheet1')

    data_with_label = []

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]

        if data_dir_name == label_list['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)
            for root, dirs, files in os.walk(path):
                for item in files:
                    if '.dcm' in item.lower():
                        # TODO 使用时加载
                        image = sitk.ReadImage(os.path.join(root, item))
                        image_array = np.squeeze(sitk.GetArrayFromImage(image))
                        data_with_label.append({'image': image_array, 'label': label_list['GOLDCLA'][i]})

    return data_with_label


if __name__ == "__main__":
    data_root_path = "/data/zengnanrong/CTDATA_test/"
    label_path = os.path.join(data_root_path, 'label_match_ct.xlsx')

    data = load_data_label(data_root_path, label_path)

    print(len(data))
