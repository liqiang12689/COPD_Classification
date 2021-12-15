import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# diabetes = pd.read_excel('./result/test_seg_cut6_30epoch.xlsx')
diabetes = pd.read_excel('./result/test_30epoch.xlsx')
fact = diabetes['label_gt']
guess = diabetes['label-pre']

print("每个类别的精确率和召回率：\n", classification_report(fact, guess))

# 混淆矩阵
classes = list(set(fact))
classes.sort()
confusion = confusion_matrix(guess, fact)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index])

plt.show()
