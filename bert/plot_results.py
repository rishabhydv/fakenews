import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import torch.optim as optim
import gc #garbage collector for gpu memory 
from tqdm import tqdm

from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from data import get_liar_dataset
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

import itertools
import numpy as np
import matplotlib.pyplot as plt

classes_6 = [0,1,2,3,4,5]
classes_2 = [0,1]
label_names_6 = (
    'true',
    'mostly-true',
    'half-true',
    'barely-true',
    'false',
    'pants-fire'
)
label_names_2 = (
    'true',
    'fake'
)
result_file = 'results/test_stats_e2.csv'
result_df = pd.read_csv(result_file, sep=",", header=None)
result = result_df.values

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

prediction = result[1:,2].flatten()
true_label = result[1:,3].flatten()
prediction_prob = result[1:,4].flatten()

for i in range(len(prediction)):
    prediction[i] = int(prediction[i][1:-1])
    true_label[i] = int(true_label[i][1:-1])
    prediction_prob[i] = float(prediction_prob[i][1:-1])

prediction = prediction.tolist()
true_label = true_label.tolist()
prediction_prob = prediction_prob.tolist()
cm = confusion_matrix(true_label,prediction)

fpr["micro"], tpr["micro"], _ = roc_curve(true_label, prediction_prob)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plot_confusion_matrix(cm,label_names_2)