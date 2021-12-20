import os
import sys

import pandas as pd
import random
import torch
from torch import nn
from torch.autograd import Variable
from dataset import load_datapath_label, load_data
from datetime import datetime
from models.densenet import densenet121

from torch.utils.tensorboard import SummaryWriter
from global_settings import CHECKPOINT_PATH, LOG_DIR, TIME_NOW

import argparse


def next_batch(batch_size, index_in_total, data, test):
    start = index_in_total
    index_in_total += batch_size
    total_num = len(data)

    # 最后一个batch
    if total_num < index_in_total < total_num + batch_size:
        index_in_total = total_num

    end = index_in_total

    batch_images = []
    batch_labels = []
    batch_dirs = []

    for i in range(start, end):
        if i < total_num:
            image_path = data[i]['image_path']
            image = load_data(image_path)
            batch_images.append(image)

            label = data[i]['label']
            batch_labels.append(label)

            if test:
                batch_dirs.append(data[i]['dir'])

    return batch_images, batch_labels, batch_dirs, index_in_total


def train(net, use_gpu, train_data, valid_data, batch_size, num_epochs, optimizer, criterion, save_model_name):
    prev_time = datetime.now()

    # use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'densenet121', TIME_NOW))

    max_vail_acc = 0.0

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    for epoch in range(num_epochs):
        random.shuffle(train_data)
        train_loss = 0.0
        train_acc = 0
        index_in_trainset = 0

        net = net.train()

        if len(train_data) % batch_size == 0:
            batch_num = int(len(train_data) / batch_size)
        else:
            batch_num = int(len(train_data) / batch_size) + 1

        for batch in range(batch_num):
            batch_images, batch_labels, _, index_in_trainset = next_batch(batch_size, index_in_trainset, train_data,
                                                                          False)
            batch_images = torch.tensor(batch_images, dtype=torch.float)

            if use_gpu:
                batch_images = Variable(torch.tensor(batch_images).cuda())
                batch_labels = Variable(torch.tensor(batch_labels).cuda())
            else:
                batch_images = Variable(torch.tensor(batch_images))
                batch_labels = Variable(torch.tensor(batch_labels))

            optimizer.zero_grad()
            output = net(batch_images)
            loss = criterion(output, batch_labels)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            _, pred_label = output.max(1)
            num_correct = pred_label.eq(batch_labels).sum()
            train_acc += num_correct

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        # 评估
        valid_loss = 0
        valid_acc = 0
        index_in_validset = 0
        net = net.eval()

        with torch.no_grad():
            if len(valid_data) % batch_size == 0:
                batch_num = int(len(valid_data) / batch_size)
            else:
                batch_num = int(len(valid_data) / batch_size) + 1

            for batch in range(batch_num):
                batch_images, batch_labels, _, index_in_validset = next_batch(batch_size, index_in_validset,
                                                                              valid_data, False)
                batch_images = torch.tensor(batch_images, dtype=torch.float)

                if use_gpu:
                    batch_images = Variable(torch.tensor(batch_images).cuda())
                    batch_labels = Variable(torch.tensor(batch_labels).cuda())
                else:
                    batch_images = Variable(torch.tensor(batch_images))
                    batch_labels = Variable(torch.tensor(batch_labels))

                output = net(batch_images)
                loss = criterion(output, batch_labels)
                valid_loss += loss.data
                _, pred_label = output.max(1)
                num_correct = pred_label.eq(batch_labels).sum()
                valid_acc += num_correct

        epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch + 1, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))

        writer.add_scalars('Loss', {'Train': train_loss / len(train_data), 'Valid': valid_loss / len(valid_data)},
                           epoch + 1)
        writer.add_scalars('Accuracy', {'Train': train_acc / len(train_data), 'Valid': valid_acc / len(valid_data)},
                           epoch + 1)

        if valid_acc / len(valid_data) > max_vail_acc:
            max_vail_acc = valid_acc / len(valid_data)
            torch.save(net, os.path.join(CHECKPOINT_PATH, save_model_name))

        prev_time = cur_time
        print(epoch_str + time_str)

    writer.close()


def test(use_gpu, test_data, batch_size, save_model_name, result_file):
    test_acc = 0
    index_in_testset = 0
    label_list = []
    outpres_list = []
    prelabels_list = []
    dirs_list = []

    net = torch.load(os.path.join(CHECKPOINT_PATH, save_model_name))
    net = net.eval()

    with torch.no_grad():
        if len(test_data) % batch_size == 0:
            batch_num = int(len(test_data) / batch_size)
        else:
            batch_num = int(len(test_data) / batch_size) + 1

        for batch in range(batch_num):
            batch_images, batch_labels, batch_dirs, index_in_testset = next_batch(batch_size, index_in_testset,
                                                                                  test_data, True)
            batch_images = torch.tensor(batch_images, dtype=torch.float)

            if use_gpu:
                batch_images = Variable(torch.tensor(batch_images).cuda())
                batch_labels = Variable(torch.tensor(batch_labels).cuda())
            else:
                batch_images = Variable(torch.tensor(batch_images))
                batch_labels = Variable(torch.tensor(batch_labels))

            output = net(batch_images)
            softmax = nn.Softmax(dim=1)
            output_softmax = softmax(output)

            _, pred_label = output.max(1)
            num_correct = pred_label.eq(batch_labels).sum()
            test_acc += num_correct

            label_list.extend(batch_labels.cpu().numpy().tolist())
            outpres_list.extend(output_softmax.cpu().numpy().tolist())
            prelabels_list.extend(pred_label.cpu().numpy().tolist())
            dirs_list.extend(batch_dirs)

        print("Test Acc: %f" % (test_acc / len(test_data)))

        df = pd.DataFrame(outpres_list, columns=['p0', 'p1', 'p2', 'p3'])
        df.insert(df.shape[1], 'label-pre', prelabels_list)
        df.insert(df.shape[1], 'label_gt', label_list)
        df.insert(df.shape[1], 'dirs', dirs_list)
        df.to_excel(result_file)


def count_person_result(input_file, output_file):
    """
    将每个病例的所有测试图像的四个等级的预测概率求平均
    :param input_file:
    :param output_file:
    :return:
    """
    input_df = pd.read_excel(input_file, sheet_name='Sheet1')
    input_df.sort_values(by='dirs', inplace=True)

    output_list = []
    count = 0
    temp_row = [0.0, 0.0, 0.0, 0.0, 0, 0, 'test']
    for i in range(len(input_df['dirs'])):
        temp_row[0] = temp_row[0] + input_df['p0'][i]
        temp_row[1] = temp_row[1] + input_df['p1'][i]
        temp_row[2] = temp_row[2] + input_df['p2'][i]
        temp_row[3] = temp_row[3] + input_df['p3'][i]
        count = count + 1

        if i + 1 < len(input_df['dirs']) and input_df['dirs'][i] is not input_df['dirs'][i + 1]:
            temp_row[0] = temp_row[0] / count
            temp_row[1] = temp_row[1] / count
            temp_row[2] = temp_row[2] / count
            temp_row[3] = temp_row[3] / count
            temp_row[4] = temp_row[:4].index(max(temp_row[:4]))
            temp_row[5] = input_df['label_gt'][i]
            temp_row[6] = input_df['dirs'][i]
            output_list.append(temp_row)

            count = 0
            temp_row = [0.0, 0.0, 0.0, 0.0, 0, 0, 'test']

        if i + 1 == len(input_df['dirs']):
            # last line
            temp_row[0] = temp_row[0] / count
            temp_row[1] = temp_row[1] / count
            temp_row[2] = temp_row[2] / count
            temp_row[3] = temp_row[3] / count
            temp_row[4] = temp_row[:4].index(max(temp_row[:4]))
            temp_row[5] = input_df['label_gt'][i]
            temp_row[6] = input_df['dirs'][i]
            output_list.append(temp_row)

    df = pd.DataFrame(output_list, columns=['p0', 'p1', 'p2', 'p3', 'label-pre', 'label_gt', 'dirs'])
    df.to_excel(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, default='/data/zengnanrong/CTDATA/', help='输入数据的根路径')
    parser.add_argument('--cut', type=bool, default=False, help='是否只截取含肺区域图像(精筛)')
    parser.add_argument('--cut_6', type=bool, default=False, help='是否只截去上下1/6的图像(粗筛)')
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否只使用GPU')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='num of epochs')
    parser.add_argument('--save_model_name', type=str, default='DenseNet121_50epoch_16batchsize.pkl',
                        help='model save name')
    parser.add_argument('--result_file', type=str, default='./result/test_50epoch_16batchsize_dir.xlsx',
                        help='test result file path')
    parser.add_argument('--cuda_device', type=str, choices=['0', '1'], default='1', help='使用哪块GPU')

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    train_valid_label_path = '/data/zengnanrong/label_match_ct_4_range_train_valid.xlsx'
    test_label_path = '/data/zengnanrong/label_match_ct_4_range_test.xlsx'

    train_valid_data_root_path = os.path.join(args.data_root_path, 'train_valid')
    test_data_root_path = os.path.join(args.data_root_path, 'test')

    train_valid_datapath_label = load_datapath_label(train_valid_data_root_path, train_valid_label_path, args.cut,
                                                     args.cut_6)
    test_datapath_label = load_datapath_label(test_data_root_path, test_label_path, args.cut, args.cut_6)
    train_data = []
    valid_data = []
    test_data = []

    for label in range(4):
        # 每个标签的数据按 训练集：验证集：测试集 6:1:3
        train_index = int(len(train_valid_datapath_label[label]) * 6 / 7)

        while train_valid_datapath_label[label][train_index]['dir'] == \
                train_valid_datapath_label[label][train_index + 1]['dir']:
            train_index = train_index + 1

        train_index = train_index + 1

        train_data.extend(train_valid_datapath_label[label][:train_index])
        valid_data.extend(train_valid_datapath_label[label][train_index:])
        test_data.extend(test_datapath_label[label])

    channels = 1
    out_features = 4  # 4分类
    pretrained = False  # 是否使用已训练模型
    drop_rate = 0.5  # 防止过拟合

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    net = densenet121(channels, out_features, args.use_gpu, pretrained, drop_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    train(net, args.use_gpu, train_data, valid_data, args.batch_size, args.num_epochs, optimizer, criterion,
          args.save_model_name)
    test(args.use_gpu, test_data, args.batch_size, args.save_model_name, args.result_file)
"""
方案一：不处理数据
 nohup python -u train.py \
 --data_root_path /data/zengnanrong/CTDATA/ \
 --cut False \
 --cut_6 False \
 --use_gpu True \
 --batch_size 20 \
 --num_epochs 50 \
 --save_model_name DenseNet121_50epoch_lr.pkl \
 --result_file ./result/test_50epoch_lr_dir.xlsx \
 --cuda_device 1 \
 > ./log/out_50epoch_lr.log &
 
 方案二：删去非肺区域的图像
  nohup python -u train.py \
 --data_root_path /data/zengnanrong/CTDATA/ \
 --cut True \
 --cut_6 False \
 --use_gpu True \
 --batch_size 20 \
 --num_epochs 50 \
 --save_model_name DenseNet121_cut_50epoch.pkl \
 --result_file ./result/test_cut_50epoch_dir.xlsx \
 --cuda_device 1 \
 > ./log/out_cut_50epoch_dir.log &
 
 方案三：提取肺实质图像_精筛
  nohup python -u train.py \
 --data_root_path /data/zengnanrong/LUNG_SEG/ \
 --cut True \
 --cut_6 False \
 --use_gpu True \
 --batch_size 20 \
 --num_epochs 50 \
 --save_model_name DenseNet121_seg_cut_50epoch.pkl \
 --result_file ./result/test_seg_cut_50epoch_dir.xlsx \
 --cuda_device 1 \
 > ./log/out_seg_cut_50epoch_dir.log &
 
 方案四：提取肺实质图像_粗筛
   nohup python -u train.py \
 --data_root_path /data/zengnanrong/LUNG_SEG/ \
 --cut False \
 --cut_6 True \
 --use_gpu True \
 --batch_size 20 \
 --num_epochs 50 \
 --save_model_name DenseNet121_seg_cut6_50epoch.pkl \
 --result_file ./result/test_seg_cut6_50epoch_dir.xlsx \
 --cuda_device 1 \
 > ./log/out_seg_cut6_50epoch_dir.log &
"""
