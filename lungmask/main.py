import sys
import argparse
import logging
import mask
import lungmask_utils
import os
import SimpleITK as sitk
import numpy as np

from pathos.multiprocessing import ProcessingPool as Pool


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main(input, output, modelpath):
    # version = pkg_resources.require("lungmask")[0].version

    parser = argparse.ArgumentParser()
    # parser.add_argument('input', metavar='input', type=path, help='Path to the input image, can be a folder for dicoms')
    # parser.add_argument('output', metavar='output', type=str, help='Filepath for output lungmask')
    parser.add_argument('--modeltype', help='Default: unet', type=str, choices=['unet'], default='unet')
    parser.add_argument('--modelname', help="spcifies the trained model, Default: R231", type=str,
                        choices=['R231', 'LTRCLobes', 'LTRCLobes_R231', 'R231CovidWeb'], default='R231')
    # parser.add_argument('--modelpath', help="spcifies the path to the trained model", default=None)
    parser.add_argument('--classes', help="spcifies the number of output classes of the model", default=3)
    parser.add_argument('--cpu', help="Force using the CPU even when a GPU is available, will override batchsize to 1",
                        action='store_true')
    parser.add_argument('--nopostprocess',
                        help="Deactivates postprocessing (removal of unconnected components and hole filling",
                        action='store_true')
    parser.add_argument('--noHU',
                        help="For processing of images that are not encoded in hounsfield units (HU). E.g. png or jpg images from the web. Be aware, results may be substantially worse on these images",
                        action='store_true')
    parser.add_argument('--batchsize', type=int,
                        help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.",
                        default=20)
    # parser.add_argument('--version', help="Shows the current version of lungmask", action='version', version=version)

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    batchsize = args.batchsize
    if args.cpu:
        batchsize = 1

    # logging.info(f'Load model')

    # input_image = lungmask_utils.get_input_image(args.input)
    input_image = lungmask_utils.get_input_image(input)
    # logging.info(f'Infer lungmask')
    if args.modelname == 'LTRCLobes_R231':
        # assert args.modelpath is None, "Modelpath can not be specified for LTRCLobes_R231 mode"
        assert modelpath is None, "Modelpath can not be specified for LTRCLobes_R231 mode"
        result = mask.apply_fused(input_image, force_cpu=args.cpu, batch_size=batchsize,
                                  volume_postprocessing=not (args.nopostprocess), noHU=args.noHU)
    else:
        # model = mask.get_model(args.modeltype, args.modelname, args.modelpath, args.classes)
        model = mask.get_model(args.modeltype, args.modelname, modelpath, args.classes)
        result = mask.apply(input_image, model, force_cpu=args.cpu, batch_size=batchsize,
                            volume_postprocessing=not (args.nopostprocess), noHU=args.noHU)

    if args.noHU:
        # file_ending = args.output.split('.')[-1]
        file_ending = output.split('.')[-1]
        print(file_ending)
        if file_ending in ['jpg', 'jpeg', 'png']:
            result = (result / (result.max()) * 255).astype(np.uint8)
        result = result[0]

    result_out = sitk.GetImageFromArray(result)
    result_out.CopyInformation(input_image)

    # output_dir = os.path.split(args.output)[0]
    output_dir = os.path.split(output)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # logging.info(f'Save result to: {args.output}')
    logging.info(f'Save result to: {output}')
    # sys.exit(sitk.WriteImage(result_out, args.output))
    sitk.WriteImage(result_out, output)


# def segment_dicoms(input_root_path, output_root_path, modelpath):
#     """
#     将input_root_path中各个文件夹中的dcm图像进行肺分割，并将分割结果按原路径格式放到output_root_path中
#     :param input_root_path:
#     :param output_root_path:
#     :param modelpath:
#     :return:
#     """
#     ct_dir = []
#     for item in os.listdir(input_root_path):
#         if os.path.isdir(os.path.join(input_root_path, item)):
#             ct_dir.append(item)
#
#     ct_dir.sort()
#
#     for i in range(len(ct_dir)):
#         data_dir_name = ct_dir[i]
#         path = os.path.join(input_root_path, data_dir_name)
#         for root, dirs, files in os.walk(path):
#             for item in files:
#                 if '.dcm' in item.lower():
#                     input = os.path.join(root, item)
#                     output = input.replace(input_root_path, output_root_path)
#                     main(input, output, modelpath)


def mask_dcms(data_dir_name):
    modelpath = '/home/MHISS/zengnanrong/COPD/checkpoint/unet_r231-d5d2fc3d.pth'
    input_root_path = "/data/zengnanrong/CTDATA/"
    output_root_path = "/data/zengnanrong/R231/"

    path = os.path.join(input_root_path, data_dir_name)
    for root, dirs, files in os.walk(path):
        for item in files:
            if '.dcm' in item.lower():
                input = os.path.join(root, item)
                output = input.replace(input_root_path, output_root_path)
                main(input, output, modelpath)


def get_ct_dirs(root_path):
    ct_dir = []
    for item in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, item)):
            ct_dir.append(item)

    ct_dir.sort()

    return ct_dir


def segement_dcms(data_dir_name):
    ct_root_path = "/data/zengnanrong/CTDATA/"
    mask_root_path = "/data/zengnanrong/R231/"
    output_root_path = "/data/zengnanrong/LUNG_SEG/"

    path = os.path.join(ct_root_path, data_dir_name)
    for root, dirs, files in os.walk(path):
        for item in files:
            if '.dcm' in item.lower():
                ct_path = os.path.join(root, item)
                mask_path = ct_path.replace(ct_root_path, mask_root_path)

                ct_image = sitk.ReadImage(ct_path)
                ct_image_array = np.squeeze(sitk.GetArrayFromImage(ct_image))
                mask_image = sitk.ReadImage(mask_path)
                mask_image_array = np.squeeze(sitk.GetArrayFromImage(mask_image))

                height = ct_image_array.shape[0]
                width = ct_image_array.shape[1]

                for h in range(height):
                    for w in range(width):
                        if mask_image_array[h][w] == 0:
                            # 将非肺区域置0
                            ct_image_array[h][w] = 0

                seg_path = ct_path.replace(ct_root_path, output_root_path)
                seg_dir = os.path.split(seg_path)[0]
                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)

                ct_image_array = np.reshape(ct_image_array, (1, height, width))
                seg_image = sitk.GetImageFromArray(ct_image_array)
                seg_image.CopyInformation(ct_image)
                sitk.WriteImage(seg_image, seg_path)
                logging.info(f'Save result to: {seg_path}')


if __name__ == "__main__":
    input_root_path = "/data/zengnanrong/CTDATA/"

    ct_dir = get_ct_dirs(input_root_path)

    pool = Pool()
    # pool.map(mask_dcms, ct_dir)
    pool.map(segement_dcms, ct_dir)
    pool.close()
    pool.join()
