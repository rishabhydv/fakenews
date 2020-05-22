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

parser = argparse.ArgumentParser(description='BERT Fake News Classification')
parser.add_argument('--data', type=str, default='data', metavar='D',help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch_size', type=int, default=16, metavar='N',help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=4, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=3e-6, metavar='LR',help='learning rate (default: 3e-6)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--workers', type=int,default=4, metavar='D',help="name of the model")
parser.add_argument('--labels', type=int,default=6, metavar='D',help="name of the model")
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = opt.batch_size
NUM_EPOCHS = opt.epochs

train_dataset, val_dataset, test_dataset = get_liar_dataset()


train_dataloader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = False
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    shuffle = False
)
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

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = 0, # Default value in run_glue.py
    num_training_steps = total_steps)
loss_function = nn.BCEWithLogitsLoss()

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(epoch):
    torch.cuda.empty_cache() #memory
    gc.collect() #memory
    total_loss = 0
    print("Start training for epoch:",epoch)
    for step,target in enumerate(train_dataloader):
        target_input_id = target[0].to(device)
        target_input_mask = target[1].to(device)
        target_labels = target[2].to(device)

        loss, logits = model(input_ids=target_input_id, attention_mask=target_input_mask, labels=target_labels)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % opt.log_interval == 0:
            avg_loss = float(total_loss / (step+1))
            print('Epoch: {} | Avg Classification Loss: {} | Memory Allocated: {}'.format(epoch, avg_loss, 0))
    print('Trained Epoch {} | Total Avg Loss: {}'.format(epoch, avg_loss))

def validation(epoch):
    torch.cuda.empty_cache() #memory
    gc.collect() #memory
    total_loss = 0
    total_eval_accuracy = 0
    print("Start validation for epoch:",epoch)
    with torch.no_grad():
        for step,target in enumerate(val_dataloader):
            target_input_id = target[0].to(device)
            target_input_mask = target[1].to(device)
            target_labels = target[2].to(device)
            
            loss, logits = model(input_ids=target_input_id, attention_mask=target_input_mask, labels=target_labels)
            total_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = target_labels.cpu().numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

            if step % opt.log_interval == 0:
                avg_loss = float(total_loss / (step+1))
                print('Epoch: {} | Avg Classification Loss: {} | Memory Allocated: {}'.format(epoch, avg_loss, 0))
        print('Trained Epoch {} | Total Avg Loss: {} | Total avg accuracy {}'.format(epoch, avg_loss,total_eval_accuracy/len(val_dataloader)))

    torch.save(model.state_dict(), 'fake_news_classification_{}.pth'.format(epoch))

for ep in range(NUM_EPOCHS):
    train(ep)
    validation(ep)