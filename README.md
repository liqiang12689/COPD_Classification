# Chronic Obstructive Pulmonary Disease(COPD) Classification

---

## 1 数据集划分

合作医院的数据集共有468例病人的CT扫描数据，按照7：3的比例，将前327例的所有图片做为train_valid_set，后141例的所有图片做为test_set。

在train_valid_set中，train_set与valid_set同样以病例为单位，按train_set：valid_set等于6：1划分。为使得train_set与valid_set中的各个COPD等级的数据量也相对均匀，先对train_valid_set中的各个COPD等级的数据进行6：1划分，最后再汇总成最终的train_set与valid_set。

最终，train_set：valid_set：test_set 为 6：1：3。

下表为train_set、valid_set、test_set 中4个COPD等级各自所含的数据量。

|           | 0               | 1               | 2               | 3               | total             |
| --------- | --------------- | --------------- | --------------- | --------------- | ----------------- |
| train_set | 76例，共43694张 | 65例，共40550张 | 75例，共43510张 | 64例，共40944张 | 280例，共168698张 |
| valid_set | 13例，共7705张  | 11例，共6981张  | 12例，共7672张  | 11例，共6924张  | 47例，共29282张   |
| test_set  | 41例，共23940张 | 33例，共21141张 | 35例，共22283张 | 32例，共20478张 | 141例，共87842张  |

## 2 Test_Set中病例COPD等级的预测

DenseNet网络模型的训练与测试，均以每张图片为单位，对病例的每张CT图像预测出4个COPD等级的概率。

对于每例测试病例，先将其所有CT图像在4个COPD等级上的预计结果分别进行求和取平均，得到每例病例在4个COPD等级上的平均概率，取数值最大的做为最终的COPD预测等级。

## 3关于DenseNet模型训练的一些问题

- [ ]  **问题一、训练过拟合**

我们对CT数据不进行任何处理，进行了一次训练。训练后发现尽管Train Accuracy达到了0.839，但Test Accuracy仅有0.418，不到Train Accuracy的一半，而且valid的Acc与Loss曲线大幅振荡，所有我们推测可能是模型训练过拟合。于是我修改了drop_rate与learning rate，又进行了第二次训练，不过根据目前训练的情况，模型好像依然过拟合，且Train Acc较第一次训练，减少了近一半。**该如何解决这一问题？**

下表为两次训练的参数与结果。

**训练参数**

|            | batch_size | epochs | drop_rate | learning rate                                   |
| ---------- | ---------- | ------ | --------- | ----------------------------------------------- |
| 第一次训练 | 20         | 50     | 0         | torch.optim.Adam(net.parameters(), lr=1e-3)     |
| 第二次训练 | 20         | 50     | **0.5**   | torch.optim.Adam(net.parameters(), **lr=3e-4**) |

**训练结果**

|                            | Acc                                                          | Loss                                                         | Test Acc |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 第一次训练                 | ![image-20211221155419330](https://s2.loli.net/2021/12/21/IOxrMRDbjpe9mKl.png) | ![image-20211221160818171](https://s2.loli.net/2021/12/21/4hXKeORcdDMwpil.png) | 0.418    |
|                            | ![image-20211221155437630](https://s2.loli.net/2021/12/21/SHcuyaxVdjrOF1T.png) | ![image-20211221160849503](https://s2.loli.net/2021/12/21/43wLklfitZudzcm.png) |          |
| 第二次训练(训练中，未结束) | ![image-20211221162057622](https://s2.loli.net/2021/12/21/KD3UrZQAWsPY4vF.png) | ![image-20211221162329450](https://s2.loli.net/2021/12/21/Lgsn8ZhwqpSTPVO.png) |          |

- [ ]  **问题二、3D DenseNet数据输入**

目前我在GitHub找到了一个3D DenseNet的实现代码：[3D-ResNets-PyTorch/densenet.py at master · kenshohara/3D-ResNets-PyTorch (github.com)](https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/densenet.py)

以及用该代码实现阿兹海默症四分类的论文：[Ruiz J., Mahmud M., Modasshir M., Shamim Kaiser M., Alzheimer’s Disease Neuroimaging Initiative,"3D DenseNet Ensemble in 4-Way Classification of Alzheimer’s Disease",Brain Informatics. BI 2020.](https://doi.org/10.1007/978-3-030-59277-6_8)

不过该论文并没有具体说明如何进行网络的3D数据输入，且在其公开的代码中（[JuanRuiz135/3D-Densenet-Alzheimer: 3D Densenet Ensemble applied in 4-way classification of Alzheimer's Disease (BI 2020) (github.com)](https://github.com/JuanRuiz135/3D-Densenet-Alzheimer)），将网络的n_input_channels设为1，这和我们目前2D DenseNet的channels是一样的，所以我就比较困惑。

3D DenseNet的输入应该是一个3维的形式，如512x512xN。最开始开会的时候，记得师兄提到过，由于每个病例的N是不统一的，有的是600多张，有的是500多张，所有需要将N统一成N'。那统一后的 512x512xN'，是一起输入吗？现在我们二维的网络训练，batch_size是20，相当于一次输入512x512x20，但N'肯定比20要大的多，512x512xN'一起输入的话应该会内存不足。

或者是参考脑组用3D Unet分割的方法，先将每例病人的512x512xN'数据进行**切块**，如切成 m * m * n的形状，将每块放入3D网络中训练，最后将每例病人所有块的预测值进行求和取平均，得到每例病人在4个COPD等级上最终的预测结果，这种方式是否可行？ 
