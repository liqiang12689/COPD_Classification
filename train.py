import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from dataset import load_datapath_label, load_data
from datetime import datetime
from models.densenet import densenet121


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = pred_label.eq(label).sum()
    return num_correct / total


def train(net, use_gpu, train_data, valid_data, num_epochs, optimizer, criterion, checkpoint_path):
    prev_time = datetime.now()

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()

        for i in range(len(train_data)):
            image_path = train_data[i]['image_path']
            image = load_data(image_path)
            image = torch.tensor(image, dtype=torch.float)

            label_int = train_data[i]['label']
            label_list = []
            label_list.append(label_int)

            if use_gpu:
                image = Variable(torch.tensor(image).cuda())
                label = Variable(torch.tensor(label_list).cuda())
            else:
                image = Variable(torch.tensor(image))
                label = Variable(torch.tensor(label_list))

            optimizer.zero_grad()
            output = net(image)
            loss = criterion(output, label)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            with torch.no_grad():
                for i in range(len(valid_data)):
                    image_path = valid_data[i]['image_path']
                    image = load_data(image_path)
                    image = torch.tensor(image, dtype=torch.float)

                    label_int = valid_data[i]['label']
                    label_list = []
                    label_list.append(label_int)

                    if use_gpu:
                        image = Variable(torch.tensor(image).cuda())
                        label = Variable(torch.tensor(label_list).cuda())
                    else:
                        image = Variable(torch.tensor(image))
                        label = Variable(torch.tensor(label_list))

                    output = net(image)
                    loss = criterion(output, label)
                    valid_loss += loss.data
                    valid_acc += get_acc(output, label)

            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch+1, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch+1, train_loss / len(train_data),
                          train_acc / len(train_data)))

        prev_time = cur_time
        print(epoch_str + time_str)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save(net, os.path.join(checkpoint_path, 'densenet121.pkl'))


if __name__ == '__main__':
    data_root_path = "/data/zengnanrong/CTDATA_test/"
    label_path = os.path.join(data_root_path, 'label_match_ct_4.xlsx')
    data = load_datapath_label(data_root_path, label_path)

    data =data[:100]

    # 训练数据与测试数据 7:3
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    valid_data = data[train_size:]

    channels = 1
    out_features = 4  # 4分类
    use_gpu = True
    pretrained = False  # 是否使用已训练模型
    num_epochs = 10
    checkpoint_path = 'checkpoint'

    net = densenet121(channels, out_features, use_gpu, pretrained)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train(net, use_gpu, train_data, valid_data, num_epochs, optimizer, criterion, checkpoint_path)
