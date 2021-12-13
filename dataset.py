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


def exist_lung(image_path):
    image = sitk.ReadImage(image_path)
    image_array = np.squeeze(sitk.GetArrayFromImage(image))
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

    label_df = pd.read_excel(os.path.join(label_path), sheet_name='Sheet1')
    label_df.insert(label_df.shape[1], 'appear_index', 0)
    label_df.insert(label_df.shape[1], 'disappear_index', 0)

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]
        if data_dir_name == label_df['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)
            for root, dirs, files in os.walk(path):
                if len(files) == 0:
                    continue

                len_files = len(files)
                files.sort()
                for appear_index in range(len_files):
                    image_path = os.path.join(root, files[appear_index])
                    if exist_lung(image_path):
                        print(i)
                        label_df['appear_index'][i] = appear_index
                        break
                for index in range(len_files):
                    disappear_index = len_files - 1 - index
                    image_path = os.path.join(root, files[disappear_index])
                    if exist_lung(image_path):
                        print(i)
                        label_df['disappear_index'][i] = disappear_index
                        break

    label_df.to_excel(output_path, index=False)


def load_datapath_label(data_root_path, label_path, cut):
    """
    加载每一张DICOM图像的路径，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :param cut: 是否截取包含肺区域的图像
    :return:
    """
    ct_dir = []
    for item in os.listdir(data_root_path):
        if os.path.isdir(os.path.join(data_root_path, item)):
            ct_dir.append(item)

    # 确保与label文件中的名称顺序对应
    ct_dir.sort()

    label_df = pd.read_excel(os.path.join(label_path), sheet_name='Sheet1')

    data_path_with_label = [[], [], [], []]

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]

        if data_dir_name == label_df['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)
            for root, dirs, files in os.walk(path):
                if len(files) == 0:
                    continue
                if cut:
                    files.sort()
                    appear_idx = label_df['appear_index'][i]
                    disappear_idx = label_df['disappear_index'][i]

                    if '.xml' in files[0].lower():
                        appear_idx = appear_idx + 1
                        disappear_idx = disappear_idx + 1

                    files = files[appear_idx:disappear_idx + 1]

                for item in files:
                    if '.dcm' in item.lower():
                        image_path = os.path.join(root, item)
                        # 训练时预测的标签范围为[0,3]
                        label = label_df['GOLDCLA'][i] - 1
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
    # label_path = os.path.join(data_root_path, 'label.xlsx')
    # output_path = os.path.join(data_root_path, 'label_match_ct_4.xlsx')
    # label_preprocess(label_path, output_path)

    # 经过lungmask Unet-R231模型分割后的肺部区域标图像
    # data_root_path = "/data/zengnanrong/R231/"
    # label_path = os.path.join(data_root_path, 'label_match_ct_4.xlsx')
    # output_path = os.path.join(data_root_path, 'label_match_ct_4_range.xlsx')
    # find_lung_range(label_path, data_root_path, output_path)

    # 分割后的肺部CT图像
    data_root_path = "/data/zengnanrong/LUNG_SEG/"
    label_path = os.path.join(data_root_path, 'label_match_ct_4_range.xlsx')
    data = load_datapath_label(data_root_path, label_path, True)
    print(len(data[0]))
    print(len(data[1]))
    print(len(data[2]))
    print(len(data[3]))
    print(len(data[0]) + len(data[1]) + len(data[2]) + len(data[3]))
