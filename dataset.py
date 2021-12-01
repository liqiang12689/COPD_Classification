import os

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

        if label_data['GOLDCLA'][i] == 5:
            # 将级别5的改为级别4，使得级别4的样本数与级别1-3的基本相同
            label_data['GOLDCLA'][i] = 4

    label_data.sort_values(by='subject')
    pd.DataFrame(label_data).to_excel(os.path.join(output_path), sheet_name='Sheet1')


def load_datapath_label(data_root_path, label_path):
    """
    加载每一张DICOM图像的路径，并为其加上对应标签
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

    data_path_with_label = []

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]

        if data_dir_name == label_list['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)
            for root, dirs, files in os.walk(path):
                for item in files:
                    if '.dcm' in item.lower():
                        image_path = os.path.join(root, item)
                        # 训练时预测的标签范围为[0,3]
                        label = label_list['GOLDCLA'][i] - 1
                        data_path_with_label.append({'image_path': image_path, 'label': label})

    return data_path_with_label


def load_data(path):
    dicom_image = sitk.ReadImage(path)
    image_array = np.squeeze(sitk.GetArrayFromImage(dicom_image))

    # TODO 参数化 batch size, channel, wide, high
    reshape_image = np.reshape(image_array, (1, 1, 512, 512))

    return reshape_image


if __name__ == "__main__":
    data_root_path = "/data/zengnanrong/CTDATA_test/"

    # label_path = os.path.join(data_root_path, 'label.xlsx')
    output_path = os.path.join(data_root_path, 'label_match_ct_4.xlsx')
    # label_preprocess(label_path, output_path)
    data = load_datapath_label(data_root_path, output_path)

    # data = data[648:652]
    # print(data)
