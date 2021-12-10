import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

diabetes = pd.read_excel('./result/test_15epoch.xlsx')
# diabetes = pd.read_excel('./result/test_seg_15epoch.xlsx')
# diabetes = pd.read_excel('./result/test_seg_15epoch_cut6.xlsx')

n_class = 4
list1 = diabetes['label_gt']
list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])

y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))

auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())

for i, value in enumerate(thersholds):
    print("%f %f %f" % (fpr[i], tpr[i], value))

plt.plot(fpr, tpr, '-', label='Densenet121_15epoch (AUC = {0:.2f})'.format(auc), lw=1)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
