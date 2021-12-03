import os

import numpy as np
import random
import torch
from torch import nn
from torch.autograd import Variable
from dataset import load_datapath_label, load_data
from datetime import datetime
from models.densenet import densenet121

from torch.utils.tensorboard import SummaryWriter
from global_settings import CHECKPOINT_PATH, LOG_DIR, TIME_NOW


def next_batch(batch_size, index_in_total, data):
    start = index_in_total
    index_in_total += batch_size
    total_num = len(data)

    # 最后一个batch
    if total_num < index_in_total < total_num + batch_size:
        index_in_total = total_num

    end = index_in_total

    batch_images = []
    batch_labels = []

    for i in range(start, end):
        image_path = data[i]['image_path']
        image = load_data(image_path)
        batch_images.append(image)

        label = data[i]['label']
        batch_labels.append(label)

    return batch_images, batch_labels, index_in_total


def train(net, use_gpu, train_data, valid_data, batch_size, num_epochs, optimizer, criterion):
    prev_time = datetime.now()

    # use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'densenet121', TIME_NOW))

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0
        index_in_trainset = 0

        net = net.train()

        if len(train_data) % batch_size == 0:
            batch_num = int(len(train_data) / batch_size)
        else:
            batch_num = int(len(train_data) / batch_size) + 1

        for batch in range(batch_num):
            batch_images, batch_labels, index_in_trainset = next_batch(batch_size, index_in_trainset, train_data)
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

        if valid_data is not None:
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
                    batch_images, batch_labels, index_in_validset = next_batch(batch_size, index_in_validset,
                                                                               valid_data)
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
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch + 1, train_loss / len(train_data),
                          train_acc / len(train_data)))

        writer.add_scalar('Train Loss', train_loss / len(train_data), epoch + 1)
        writer.add_scalar('Train Acc', train_acc / len(train_data), epoch + 1)
        if valid_data is not None:
            writer.add_scalar('Valid loss', valid_loss / len(valid_data), epoch + 1)
            writer.add_scalar('Valid Acc', valid_acc / len(valid_data), epoch + 1)

        prev_time = cur_time
        print(epoch_str + time_str)

    writer.close()

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    torch.save(net, os.path.join(CHECKPOINT_PATH, 'densenet121.pkl'))


if __name__ == '__main__':
    # data_root_path = "/data/zengnanrong/CTDATA_test/"
    data_root_path = "/data/zengnanrong/CTDATA/"
    label_path = os.path.join(data_root_path, 'label_match_ct_4.xlsx')

    # len(data) = 255301
    data = load_datapath_label(data_root_path, label_path)

    random.shuffle(data)

    data = data[:1000]

    # 训练数据与测试数据 7:3
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    valid_data = data[train_size:]

    channels = 1
    out_features = 4  # 4分类
    use_gpu = True
    pretrained = False  # 是否使用已训练模型
    batch_size = 20
    num_epochs = 100

    net = densenet121(channels, out_features, use_gpu, pretrained)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train(net, use_gpu, train_data, valid_data, batch_size, num_epochs, optimizer, criterion)
