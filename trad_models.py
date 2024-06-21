##STEP 1: Open all files

import pandas as pd

anime_train = pd.read_csv(".//Database//Anime_train.csv")
anime_test = pd.read_csv(".//Database//Anime_test.csv")

char_train = pd.read_csv(".//Database//Char_train.csv")
char_test = pd.read_csv(".//Database//Char_test.csv")

impath_train = ".//Database//char train//"
impath_test = ".//Database//char test//"


#Original format for characters:
anime_train["Main Characters"] = anime_train["Main Characters"].apply(lambda x:[int(y) for y in x[1:-1].split(", ")])
anime_test["Main Characters"] = anime_test["Main Characters"].apply(lambda x:[int(y) for y in x[1:-1].split(", ")])


###ALL GENERAL VALUES:

batch_size = 16
max_length_syn = 128
max_length_char = 256
epochs = 30 #1000

learning_rate = 5e-5
eps_value = 1e-8
syn_model_name_or_path = 'gpt2'
char_model_name_or_path = 'gpt2'
img_model_name_or_path = 'microsoft/resnet-50'

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
    def __init__(self, df, char_df, mchars, img_path, transform=None):

        self.labels = []
        for x in df.index:
            self.labels.append(df["Score"][x])
        
        self.syn = []
        for x in df.index:
            self.syn.append(df["Synopsis"][x].split("(source")[0].split("(Source")[0].split("[Written")[0])

        self.mchars = []
        for x in df.index:
            list_chars = mchars[x]
            chars = []
            for y in list_chars:
                personaje = char_df[char_df["MAL ID"] == y].reset_index()
                chars.append(personaje["Description"][0].split("(source")[0].split("(Source")[0].split("[Written")[0])
            self.mchars.append(' '.join(chars))

        self.transform = transform
        self.img = []
        for x in tqdm(df.index, desc="Concatenating portraits"):
            list_chars = [str(y)+".png" for y in mchars[x]]
            chars = []
            for y in list_chars:
                img_name = img_path+y
                personaje = Image.open(img_name).convert('RGB') #Abrir la imagen
                chars.append(personaje)

            #Appending all portraits, horizontally.
            widths, heights = zip(*(img.size for img in chars))
            retratos = Image.new('RGB', (sum(widths), max(heights)))
            
            x_offset = 0
            for img in chars:
                retratos.paste(img, (x_offset, 0))
                x_offset += img.width

            if self.transform:
                retratos = self.transform(retratos)

            self.img.append(retratos)

        self.labels = scaler(self.labels, 1)

        return
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {'synopsis':self.syn[item], 'mchar':self.mchars[item], 'img':self.img[item], 'label':self.labels[item]}
    

transf = None
####It can be:
##
##transf = transforms.Compose([
##    transforms.Resize(size),        # You should first define a tuple "size". This can aid in memory saving.
##    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images. Values are just illustrative.
##    ])

train_dataset = AnimeDataset(df=anime_train, char_df=char_train, mchars=anime_train["Main Characters"], img_path=impath_train, transform=transf)
test_dataset = AnimeDataset(df=anime_test, char_df=char_test, mchars=anime_test["Main Characters"], img_path=impath_test, transform=transf)


import re
from sklearn.feature_extraction.text import TfidfVectorizer

class tfidf():
    def __init__(self):
        self.vec = TfidfVectorizer()

    def prepros(self, texto):
        return re.sub(r'[^A-Za-z0-9\s]', ' ', texto)
    
    def vectorizer(self, texts):
        pad = 750
        prepros_docs = [self.prepros(x) for x in texts]
        x = self.vec.fit_transform(prepros_docs).toarray()

        padding = torch.zeros(x.shape[0], pad)
        x_tensor = torch.cat((torch.Tensor(x), padding), dim=1)[:,:pad]
        return x_tensor
    

import torchvision.transforms as transforms

class img_tensorification():
    def __init__(self):
        self.transform = transforms.Compose([transforms.PILToTensor()])
    
    def vectorizer(self, imgs):
        pad = 750
        padding = torch.zeros(1, pad)
        img_tensors = [torch.cat((self.transform(im).reshape(1, -1), padding), dim=1)[:, :pad] for im in imgs]
        img_tensors = torch.cat(img_tensors, dim=0)
        return img_tensors
    

###STEP 3: Collate all the data into a unified input

class AnimeRegressionCollator(object):
    def __init__(self, img_vectorizer, syn_vectorizer, char_vectorizer):
        self.syn_vectorizer = syn_vectorizer
        self.char_vectorizer = char_vectorizer
        self.img_vectorizer = img_vectorizer
        return

    def __call__(self, sequences):
        labels = [sequence['label'] for sequence in sequences]

        synopsis = [sequence['synopsis'] for sequence in sequences]
        char = [sequence['mchar'] for sequence in sequences]
        img = [sequence['img'] for sequence in sequences]

        syn_v = self.syn_vectorizer
        char_v = self.char_vectorizer
        img_v = self.img_vectorizer

        inputs = {'portraits': img_v.vectorizer(img)}
        inputs.update({'synopsis': syn_v.vectorizer(synopsis)})
        inputs.update({'descriptions': char_v.vectorizer(char)})
        inputs.update({'labels': torch.tensor(np.array(labels), dtype=torch.float)})
        return inputs
    


img_v = img_tensorification()
syn_v = tfidf()
char_v = tfidf()

regression_collator = AnimeRegressionCollator(img_vectorizer= img_tensorification(), 
                                              syn_vectorizer= tfidf(), 
                                              char_vectorizer= tfidf())

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=regression_collator)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=regression_collator)


###STEP 4: Initialize the neural network

import torch.nn as nn
from transformers import GPT2Model, ResNetModel

class TradNet(nn.Module):
    def __init__(self, pad):
        super(TradNet, self).__init__()
        self.pad = pad

        #CHARACTERS:
        self.linear_regression = nn.Sequential(
                nn.Linear(self.pad,1000, bias=True),
                nn.Tanh(),
                nn.Linear(1000,500, bias=True),
                nn.Tanh(),
                nn.Linear(500,250, bias=True),
                nn.Tanh(),
                nn.Linear(250,100, bias=True),
                nn.Tanh(),
                nn.Linear(100,1, bias=True),
                nn.Softmax(dim=0)
        )
        
    def forward(self, syn_input_ids, char_input_ids, img_input_ids):

        #print(syn_input_ids.shape, char_input_ids.shape, img_input_ids.shape)

        all = torch.concat((syn_input_ids, char_input_ids, img_input_ids), dim=1)
        if all.shape[1] >= self.pad:
            all = all[:, :self.pad]
        else:
            padding = torch.zeros(1, self.pad)
            all = torch.cat((all, padding), dim=1)[:, :self.pad]

        logits = self.linear_regression(all)

        return logits
    

model = TradNet(pad=2250)
model.to(device)



def training_loop(train_loader, predictions, true_labels, optimizer_, scheduler_, device_, loss_fn):
    global model

    model.train()

    total_loss = 0

    for batch in tqdm(train_loader, total=len(train_loader), desc="Batch"):

        true_labels += batch['labels'].numpy().flatten().tolist()

        model.zero_grad()
        
        ##INPUTS:
        
        syn_input_ids = batch['synopsis'].type(torch.long).to(device_)
        char_input_ids = batch['descriptions'].type(torch.long).to(device_)

        img_input_ids = batch['portraits'].type(torch.float).to(device_)

        
        logits = model(syn_input_ids=syn_input_ids, char_input_ids=char_input_ids, img_input_ids=img_input_ids).to(device_)

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
        syn_input_ids = batch['synopsis'].type(torch.long).to(device_)
        char_input_ids = batch['descriptions'].type(torch.long).to(device_)

        img_input_ids = batch['portraits'].type(torch.float).to(device_)

        with torch.no_grad(): # Telling the model not to compute or store gradients, saving memory and speeding up validation
            outputs = model(syn_input_ids=syn_input_ids, char_input_ids=char_input_ids, img_input_ids=img_input_ids).to(device_)
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
    reps.to_csv(".//final_reports//trad//reporte_full.csv")

    for x in [[valid_labels[i], valid_predict[i]] for i in range(10)]:
        print(x)

    print("  train_loss: %.5f - val_loss: %.5f "%(train_loss, val_loss))
    print()
    trainval.append([train_loss, val_loss])


import json

jsonfile = open(".//final_reports//trad//train_val_loss.json", "w")
json.dump(trainval, jsonfile)
jsonfile.close()