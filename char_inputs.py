##STEP 1: Open all files

import pandas as pd

anime_train = pd.read_csv(".//Database//Anime_train.csv")
anime_test = pd.read_csv(".//Database//Anime_test.csv")

char_train = pd.read_csv(".//Database//Char_train.csv")
char_test = pd.read_csv(".//Database//Char_test.csv")

#Original format for characters:
anime_train["Main Characters"] = anime_train["Main Characters"].apply(lambda x:[int(y) for y in x[1:-1].split(", ")])
anime_test["Main Characters"] = anime_test["Main Characters"].apply(lambda x:[int(y) for y in x[1:-1].split(", ")])


###ALL GENERAL VALUES:

batch_size = 16
max_length_char = 256
epochs = 5 #1000

learning_rate = 5e-5
eps_value = 1e-8
char_model_name_or_path = 'gpt2'

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"torch running in {device}")

from transformers import set_seed
set_seed(42)


###STEP 2: Open all datasets

from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image 
import numpy as np

from tqdm import tqdm

def scaler(values, ratio): #Only works with positive values.
    return [(v - min(values)) * (ratio / (max(values) - min(values))) for v in values]

class AnimeDataset(Dataset):
    def __init__(self, df, char_df, mchars):

        self.labels = []
        for x in df.index:
            self.labels.append(df["Score"][x])
        
        self.mchars = []
        for x in df.index:
            list_chars = mchars[x]
            chars = []
            for y in list_chars:
                personaje = char_df[char_df["MAL ID"] == y].reset_index()
                chars.append(personaje["Description"][0].split("(source")[0].split("(Source")[0].split("[Written")[0])
            self.mchars.append(' '.join(chars))

        self.labels = scaler(self.labels, 1)

        return
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {'mchar':self.mchars[item], 'label':self.labels[item]}
    

train_dataset = AnimeDataset(df=anime_train, char_df=char_train, mchars=anime_train["Main Characters"])
test_dataset = AnimeDataset(df=anime_test, char_df=char_test, mchars=anime_test["Main Characters"])


###STEP 3: Collate all the data into a unified input

class AnimeRegressionCollator(object):
    def __init__(self, char_tokenizer, char_max_sequence_len=None):
        self.char_tokenizer = char_tokenizer
        self.char_max_sequence_len = char_tokenizer.model_max_length if char_max_sequence_len is None else char_max_sequence_len
        return

    def __call__(self, sequences):
        labels = [sequence['label'] for sequence in sequences]

        char = [sequence['mchar'] for sequence in sequences]

        inputs = {'descriptions': self.char_tokenizer(text=char, return_tensors="pt", padding=True, truncation=True, max_length=self.char_max_sequence_len)}
        inputs.update({'labels': torch.tensor(np.array(labels), dtype=torch.float)})
        return inputs


from transformers import AutoTokenizer

char_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=char_model_name_or_path)
char_tokenizer.padding_side = "left" # default to left padding
char_tokenizer.pad_token = char_tokenizer.eos_token # Define PAD Token = EOS Token = 50256

regression_collator = AnimeRegressionCollator(char_tokenizer=char_tokenizer, char_max_sequence_len=max_length_char)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=regression_collator)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=regression_collator)


###STEP 4: Initialize the neural network

import torch.nn as nn
from transformers import GPT2Model

class AniNet(nn.Module):
    def __init__(self, pretr_b):
        super(AniNet, self).__init__()

        #CHARACTERS:
        self.model_b = GPT2Model.from_pretrained(pretr_b)

        #ALL:
        self.final_classifier = nn.Sequential(
                nn.Linear(768, 384),
                nn.Tanh(),
                nn.Linear(384, 192),
                nn.Tanh(),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 12),
                nn.ReLU(),
                nn.Linear(12, 6),
                nn.ReLU(),
                nn.Linear(6, 1)
                )
        
    def forward(self, char_input_ids, char_attention_mask):

        #Characters:
        logits_b = self.model_b(char_input_ids, attention_mask=char_attention_mask).last_hidden_state[:, 0, :]
        
        #ALL:
        output = self.final_classifier(logits_b)

        return output
    

model = AniNet(pretr_b=char_model_name_or_path)
model.to(device)



def training_loop(train_loader, predictions, true_labels, optimizer_, scheduler_, device_, loss_fn):
    global model

    model.train()

    total_loss = 0

    for batch in tqdm(train_loader, total=len(train_loader), desc="Batch"):

        true_labels += batch['labels'].numpy().flatten().tolist()

        model.zero_grad()
        
        ##INPUTS:
        char_input_ids = batch['descriptions']['input_ids'].type(torch.long).to(device_)
        char_attention_mask = batch['descriptions']['attention_mask'].type(torch.long).to(device_)

        outputs = model(char_input_ids=char_input_ids, char_attention_mask=char_attention_mask).to(device_)
        
        logits = outputs

        predictions_loss = logits.squeeze()

        lbels = torch.Tensor(batch['labels'].float()).to(device_)
        loss = loss_fn(predictions_loss, lbels)
        total_loss += loss.item()

        #optimizer_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()

        predictions += predictions_loss

    avg_epoch_loss = total_loss / len(train_loader)
    return true_labels, predictions, avg_epoch_loss


def validation(test_loader, device_, loss_fn):
    global model

    predictions = []
    true_labels = []

    total_loss = 0

    model.eval()

    for batch in tqdm(test_loader, total=len(test_loader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        
        ##INPUTS:
        char_input_ids = batch['descriptions']['input_ids'].type(torch.long).to(device_)
        char_attention_mask = batch['descriptions']['attention_mask'].type(torch.long).to(device_)


        with torch.no_grad(): # Telling the model not to compute or store gradients, saving memory and speeding up validation
            outputs = model(char_input_ids=char_input_ids, char_attention_mask=char_attention_mask).to(device_)
            logits = outputs

            predictions += logits.squeeze().detach().cpu().tolist()
            predictions_loss = torch.Tensor(logits.squeeze().detach().cpu()).to(device_)

            loss = loss_fn(predictions_loss, torch.Tensor(batch['labels'].float()).to(device_))

            total_loss += loss.item()

    avg_epoch_loss = total_loss / len(test_loader)

    return true_labels, predictions, avg_epoch_loss



from Ca_Naxca import regression_report
from transformers import get_linear_schedule_with_warmup

##TRAIN THE MODEL.

optimizer_ = torch.optim.AdamW(model.parameters(), lr = learning_rate, eps = eps_value)
total_steps = len(train_dataloader) * epochs
scheduler_ = get_linear_schedule_with_warmup(optimizer_, num_warmup_steps = 0, num_training_steps = total_steps)
loss_fn = nn.MSELoss()  # Loss function for regression problems


trainval = []

for epoch in tqdm(range(epochs), desc="Epoch"):
    true_labels = []
    predictions = []
    avg_epoch_loss = 0

    # Train the model:
    true_labels, predictions, train_loss = training_loop(train_dataloader, predictions, true_labels, optimizer_, scheduler_, device, loss_fn)
    # Test the model:
    valid_labels, valid_predict, val_loss = validation(test_dataloader, device, loss_fn)

    # Is it good enough?
    reporte = regression_report(valid_labels, valid_predict, [i for i in range(len(valid_labels))])
    reps = reporte.display()
    print(reps)
    reps.to_csv(".//final_reports//Char//reporte_full.csv")

    for x in [[valid_labels[i], valid_predict[i]] for i in range(10)]:
        print(x)

    print("  train_loss: %.5f - val_loss: %.5f "%(train_loss, val_loss))
    print()
    trainval.append([train_loss, val_loss])


import json

jsonfile = open(".//final_reports//Char//train_val_loss.json", "w")
json.dump(trainval, jsonfile)
jsonfile.close()