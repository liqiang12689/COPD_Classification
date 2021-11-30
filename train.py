import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from dataset import load_data_label
from datetime import datetime
from models.densenet import densenet121


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().data[0]
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()

    prev_time = datetime.now()

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()

        for i in range(len(train_data)):
            image = train_data[i]['image']
            tempdata = np.reshape(image, (1, 1, 512, 512))
            image = torch.tensor(tempdata, dtype=torch.float)

            label = train_data[i]['label']

            if torch.cuda.is_available():
                image = Variable(torch.tensor(image).cuda())
                label = Variable(torch.tensor(label).cuda())
            else:
                image = Variable(image)
                label = Variable(label)

            output = net(image)
            loss = criterion(output, label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for i in range(len(valid_data)):
                image = valid_data[i]['image']
                tempdata = np.reshape(image, (1, 1, 512, 512))
                image = torch.tensor(tempdata, dtype=torch.float)

                label = valid_data[i]['label']

                if torch.cuda.is_available():
                    image = Variable(torch.tensor(image).cuda())
                    label = Variable(torch.tensor(label).cuda())
                else:
                    image = Variable(image)
                    label = Variable(label)

                output = net(image)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))

        prev_time = cur_time
        print(epoch_str + time_str)

    torch.save(net, os.path.join('models', 'densenet121.pkl'))


if __name__ == '__main__':
    data_root_path = "/data/zengnanrong/CTDATA_test/"
    label_path = os.path.join(data_root_path, 'label_match_ct.xlsx')
    data = load_data_label(data_root_path, label_path)

    # 训练数据与测试数据 7:3
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    valid_data = data[train_size:]

    channels = 1
    out_features = 5  # 5分类
    use_gpu = True
    pretrained = False  # 是否使用已训练模型

    net = densenet121(channels, out_features, use_gpu, pretrained)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train(net, train_data, valid_data, 10, optimizer, criterion)
