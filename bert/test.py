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

from data import get_liar_dataset, get_gpt_test_dataset
#from sklearn.metrics import plot_confusion_matrix, confusion_matrix

parser = argparse.ArgumentParser(description='BERT Fake News Classification- TEST')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--workers', type=int,default=4, metavar='D',help="name of the model")
parser.add_argument('--labels', type=int,default=2, metavar='D',help="name of the model")
parser.add_argument('--use_gpt_data', type=int,default=0, metavar='D',help="name of the model")
parser.add_argument('--model_path', type=str,default='model/fake_news_e2_c2_classification_3.pth', metavar='D',help="name of the model")
parser.add_argument('--test_stats', type=str,default='test_result', metavar='D',help="name of the model")
parser.add_argument('--results_dir', type=str,default='results/', metavar='D',help="name of the model")
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = opt.batch_size
NUM_EPOCHS = opt.epochs

if opt.use_gpt_data == 0:
    test_dataset = get_liar_dataset()
else:
    test_dataset = get_gpt_test_dataset()

test_dataloader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False
)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-cased',
    num_labels = opt.labels, # The number of output labels--2 for binary classification. You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False)
model.load_state_dict(torch.load(opt.model_path, map_location=device))
model.to(device)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds.numpy(), axis=1).flatten()
    labels_flat = labels.numpy().flatten()
    pred_softmax = nn.functional.softmax(preds)
    pred_value,pred_position = torch.max(pred_softmax,-1)
    return np.sum(pred_position.numpy() == labels_flat) / len(labels_flat), pred_position.numpy(), labels_flat, pred_value.numpy()

def validation(epoch):
    torch.cuda.empty_cache() #memory
    gc.collect() #memory
    total_loss = 0
    total_eval_accuracy = 0
    loss_data = []
    prediction = []
    prediction_probality = []
    true_label = []
    accuracy = []
    print("Start validation for epoch:",epoch)
    with torch.no_grad():
        for step,target in enumerate(test_dataloader):
            target_input_id = target[0].to(device)
            target_input_mask = target[1].to(device)
            target_labels = target[2].to(device)
            
            loss, logits = model(input_ids=target_input_id, attention_mask=target_input_mask, labels=target_labels)
            total_loss += loss.item()

            logits = logits.detach().cpu()
            label_ids = target_labels.cpu()
            acc, pred, true_val, pred_val = flat_accuracy(logits, label_ids)
            total_eval_accuracy += acc
            loss_data.append(loss.item())
            prediction.append(pred)
            true_label.append(true_val)
            prediction_probality.append(pred_val)
            accuracy.append(acc)
            if step % opt.log_interval == 0:
                avg_loss = float(total_loss / (step+1))
                print('Epoch: {} | Avg Classification Loss: {} | Memory Allocated: {}'.format(epoch, avg_loss, 0))

        test_stats = {
            'loss_data': loss_data,
            'prediction': prediction,
            'true_label': true_label,
            'prediction_prob': prediction_probality,
            'accuracy': accuracy,
            'total_accuracy': total_eval_accuracy/len(test_dataloader)
        }
        print('Trained Epoch {} | Total Avg Loss: {} | Total avg accuracy {}'.format(epoch, avg_loss,total_eval_accuracy/len(test_dataloader)))
        return test_stats


test_stats = validation(0)
df = pd.DataFrame(test_stats)
df.to_csv(opt.results_dir + opt.test_stats)