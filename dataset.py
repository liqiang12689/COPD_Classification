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


def exist_lung(image):
    image_array = image[0]
    # 肺部区域从图像的中间开始出现，可以从第100行开始扫
    for x in range(100, len(image_array[0])):
        for y in range(len(image_array[1])):
            if image_array[x][y] > 0:
                return True
    return False


def find_lung_range(label_path, data_root_path, output_path):
    """
    读取data_root_path中的CT图像，确定每例病人的含肺图像的范围，并将最先出现的和最后消失的含肺图像索引存入在output_path的excel文件中
    :param label_path:
    :param data_root_path:
    :return:
    """
    ct_dir = []
    for item in os.listdir(data_root_path):
        if os.path.isdir(os.path.join(data_root_path, item)):
            ct_dir.append(item)

    # 确保与label文件中的名称顺序对应
    ct_dir.sort()

    label_list = pd.read_excel(os.path.join(label_path), sheet_name='Sheet1')

    lung_appear_index_list = []
    lung_disappear_index_list = []

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]
        if data_dir_name == label_list['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)
            for root, dirs, files in os.walk(path):
                if len(files) == 0:
                    continue

                len_files = len(files)
                files.sort()
                for appear_index in range(len_files):
                    image_path = os.path.join(root, files[appear_index])
                    image_array = load_data(image_path)
                    if exist_lung(image_array):
                        lung_appear_index_list.append(appear_index)
                        print("appear:%s  /  %d" % (data_dir_name, appear_index))
                        break
                for index in range(len_files):
                    disappear_index = len_files - 1 - index
                    image_path = os.path.join(root, files[disappear_index])
                    image_array = load_data(image_path)
                    if exist_lung(image_array):
                        lung_disappear_index_list.append(disappear_index)
                        print("disappear:%s  /  %d" % (data_dir_name, disappear_index))
                        break

    df = pd.DataFrame(label_list)
    df.insert(df.shape[1], 'appear_index', lung_appear_index_list)
    df.insert(df.shape[1], 'disappear_index', lung_disappear_index_list)
    df.to_excel(output_path, index=False)


def load_datapath_label(data_root_path, label_path, cut):
    """
    加载每一张DICOM图像的路径，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :param cut: 是否忽略上下1/6,截取主要包含肺区域的图像
    :return:
    """
    ct_dir = []
    for item in os.listdir(data_root_path):
        if os.path.isdir(os.path.join(data_root_path, item)):
            ct_dir.append(item)

    # 确保与label文件中的名称顺序对应
    ct_dir.sort()

    label_list = pd.read_excel(os.path.join(label_path), sheet_name='Sheet1')

    data_path_with_label = [[], [], [], []]

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]

        if data_dir_name == label_list['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)
            for root, dirs, files in os.walk(path):
                if cut:
                    files.sort()
                    # TODO

                    # 忽略上下1/6,截取主要包含肺区域的图像
                    # files = files[cut_index_head:cut_index_rear]

                for item in files:
                    if '.dcm' in item.lower():
                        image_path = os.path.join(root, item)
                        # 训练时预测的标签范围为[0,3]
                        label = label_list['GOLDCLA'][i] - 1
                        data_path_with_label[label].append({'image_path': image_path, 'label': label})

    return data_path_with_label


def load_data(path):
    dicom_image = sitk.ReadImage(path)
    # image_array.shape = (1,512,512)
    image_array = sitk.GetArrayFromImage(dicom_image)

    return image_array


if __name__ == "__main__":
    # 肺部CT原始图像
    # data_root_path = "/data/zengnanrong/CTDATA/"
    # 经过lungmask Unet-R231模型分割后的肺部区域标图像
    data_root_path = "/data/zengnanrong/R231/"
    # 分割后的肺部CT图像
    # data_root_path = "/data/zengnanrong/LUNG_SEG/"

    label_path = os.path.join(data_root_path, 'label_match_ct_4.xlsx')
    output_path = os.path.join(data_root_path, 'label_match_ct_4_range.xlsx')

    find_lung_range(label_path, data_root_path, output_path)
    # data = load_datapath_label(data_root_path, label_path)
    # print(len(data))
