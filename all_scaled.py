##STEP 1: Open all files

import pandas as pd

anime_train = pd.read_csv(".//Database//Anime_train.csv")#.head(500)
anime_test = pd.read_csv(".//Database//Anime_test.csv")#.head(10)

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
epochs = 5 #1000

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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



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
    def __init__(self, df, char_df, mchars, img_path, transform=None):#, img_size):

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

            #retratos = retratos.resize(img_size) #Resize it.
            self.img.append(retratos)

        #global scaler

        # Rescaling the labels for better results in regression:
        #scaler.fit(np.array(self.labels).reshape(-1, 1))
        #self.labels = list(scaler.transform(np.array(self.labels).reshape(-1, 1)))

        self.labels = scaler(self.labels, 1)

        return
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {'synopsis':self.syn[item], 'mchar':self.mchars[item], 'img':self.img[item], 'label':self.labels[item]}


#from torchvision import transforms

transf = None #transforms.Compose([
#    transforms.Resize(size),        # Resize images to a fixed size
#    transforms.ToTensor(),          # Convert images to tensors
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
#    ])

train_dataset = AnimeDataset(df=anime_train, char_df=char_train, mchars=anime_train["Main Characters"], img_path=impath_train, transform=transf)
test_dataset = AnimeDataset(df=anime_test, char_df=char_test, mchars=anime_test["Main Characters"], img_path=impath_test, transform=transf)



###STEP 3: Collate all the data into a unified input

class AnimeRegressionCollator(object):
    def __init__(self, img_processor, syn_tokenizer, char_tokenizer, syn_max_sequence_len=None, char_max_sequence_len=None):
        self.syn_tokenizer = syn_tokenizer
        self.char_tokenizer = char_tokenizer
        self.syn_max_sequence_len = syn_tokenizer.model_max_length if syn_max_sequence_len is None else syn_max_sequence_len
        self.char_max_sequence_len = char_tokenizer.model_max_length if char_max_sequence_len is None else char_max_sequence_len
        self.img_processor = img_processor
        return

    def __call__(self, sequences):
        labels = [sequence['label'] for sequence in sequences]

        synopsis = [sequence['synopsis'] for sequence in sequences]
        char = [sequence['mchar'] for sequence in sequences]
        img = [sequence['img'] for sequence in sequences]

        inputs = {'portraits': self.img_processor(images=img, return_tensors="pt")}
        inputs.update({'synopsis': self.syn_tokenizer(text=synopsis, return_tensors="pt", padding=True, truncation=True, max_length=self.syn_max_sequence_len)})
        inputs.update({'descriptions': self.char_tokenizer(text=char, return_tensors="pt", padding=True, truncation=True, max_length=self.char_max_sequence_len)})
        inputs.update({'labels': torch.tensor(np.array(labels), dtype=torch.float)})
        return inputs

#inputs = {'input_ids': torch.tensor(np.array(imgs)), 'labels':torch.tensor(np.array(labels), dtype=torch.float)}
    

from transformers import AutoTokenizer
from transformers import AutoImageProcessor

syn_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=syn_model_name_or_path)
syn_tokenizer.padding_side = "left" # default to left padding
syn_tokenizer.pad_token = syn_tokenizer.eos_token # Define PAD Token = EOS Token = 50256

char_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=char_model_name_or_path)
char_tokenizer.padding_side = "left" # default to left padding
char_tokenizer.pad_token = char_tokenizer.eos_token # Define PAD Token = EOS Token = 50256

processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path=img_model_name_or_path)

regression_collator = AnimeRegressionCollator(img_processor=processor,
                                            syn_tokenizer=syn_tokenizer, 
                                            char_tokenizer=char_tokenizer, 
                                            syn_max_sequence_len=max_length_syn, 
                                            char_max_sequence_len=max_length_char)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=regression_collator)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=regression_collator)




###STEP 4: Initialize the neural network

import torch.nn as nn
from transformers import GPT2Model, ResNetModel

class AniNet(nn.Module):
    def __init__(self, pretr_a, pretr_b, pretr_img): #img_input_size
        super(AniNet, self).__init__()
        #self.img_input_size = img_input_size

        #CHARACTERS:
        self.img_net = ResNetModel.from_pretrained(pretr_img)
        self.model_b = GPT2Model.from_pretrained(pretr_b)

        self.char_classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(768+49, 768, bias=True),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(768, 768, bias=True)
                )
        
        #SYNOPSIS:
        self.model_a = GPT2Model.from_pretrained(pretr_a)

        #ALL:
        self.final_classifier = nn.Sequential(
                #nn.Dropout(p=0.1),
                nn.Linear(2*768,768),
                nn.Tanh(),
                nn.Linear(768, 384),
                nn.Tanh(),
                nn.Linear(384, 192),
                nn.Tanh(),
                #nn.Dropout(p=0.1),
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

    def forward(self, syn_input_ids, char_input_ids, syn_attention_mask, char_attention_mask, img_input_ids):

        #print(syn_input_ids.shape)
        #print(char_input_ids.shape)

        #Synopsis:
        logits_a = self.model_a(syn_input_ids, attention_mask=syn_attention_mask).last_hidden_state[:, 0, :]

        #Characters:
        logits_b = self.model_b(char_input_ids, attention_mask=char_attention_mask).last_hidden_state[:, 0, :]
        img_output = self.img_net(img_input_ids).last_hidden_state[:, 0, :]

        img_output = img_output.view(img_output.shape[0], -1) #Flattening to shape [bsize, 49]

        char = torch.concat((logits_b, img_output), dim=1)
        char = self.char_classifier(char)

        #ALL:
        concatenated_vectors = torch.concat((logits_a, char), dim=1)
        output = self.final_classifier(concatenated_vectors)

        ##print(output.shape) [16,1]
        
        #Final Classification:
        #output = output.view(-1)
        
        #print("#########", output)

        #output = x.view(-1)

        #output = torch.sigmoid(output)
        #output = output * 5 + 5 #[5,10]
        #output = torch.round(output)
        #output = torch.clamp(output, 5, 9)

        return output
    

model = AniNet(pretr_a=syn_model_name_or_path, pretr_b=char_model_name_or_path, pretr_img=img_model_name_or_path)
model.to(device)


#def longear(batch, device_):
#    return {k:v.type(torch.long).to(device_) for k,v in batch.items()}



def training_loop(train_loader, predictions, true_labels, optimizer_, scheduler_, device_, loss_fn):
    global model

    model.train()

    total_loss = 0

    for batch in tqdm(train_loader, total=len(train_loader), desc="Batch"):

        true_labels += batch['labels'].numpy().flatten().tolist()

        #batch = {'portraits':batch['portraits'], 'synopsis':batch['synopsis'], 'descriptions':batch['descriptions']} #Remove labels
        #batch = {k:longear(v,device_) for k,v in batch.items()}
        #batch['labels'] = leibs #Recover labels

        model.zero_grad()
        
        ##INPUTS:
        syn_input_ids = batch['synopsis']['input_ids'].type(torch.long).to(device_)
        syn_attention_mask = batch['synopsis']['attention_mask'].type(torch.long).to(device_)

        char_input_ids = batch['descriptions']['input_ids'].type(torch.long).to(device_)
        char_attention_mask = batch['descriptions']['attention_mask'].type(torch.long).to(device_)

        img_key = 'pixel_values' #'input_ids'
        img_input_ids = batch['portraits'][img_key].type(torch.float).to(device_)

        #print("##################", img_input_ids.shape)

        
        outputs = model(syn_input_ids=syn_input_ids, char_input_ids=char_input_ids, syn_attention_mask=syn_attention_mask, char_attention_mask=char_attention_mask, img_input_ids=img_input_ids).to(device_)
        
        logits = outputs

        ##print("###", logits.shape) [16,1]

        predictions_loss = logits.squeeze()

        #predictions += logits.squeeze().detach().cpu()
        #predictions_loss = torch.mean(logits, dim=0, keepdim=True).to(device_)
        #torch.Tensor(logits.squeeze().detach().cpu()).requires_grad_().to(device_) #torch.mean(logits, dim=0, keepdim=True).to(device_) #dim=1

        ##print("##", predictions.shape) [1,1]

        lbels = torch.Tensor(batch['labels'].float()).to(device_)
        loss = loss_fn(predictions_loss, lbels) #torch.nn.functional.mse_loss(predictions, batch['labels'].float()) # Loss function for regression problems
        #print(loss)
        total_loss += loss.item()


        #optimizer_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()

        predictions += predictions_loss
        #print(len(predictions))
        ##predictions = predictions.tolist()

    avg_epoch_loss = total_loss / len(train_loader)
    return true_labels, predictions, avg_epoch_loss


def validation(test_loader, device_, loss_fn):
    global model

    # Tracking variables
    predictions = []
    true_labels = []

    total_loss = 0

    model.eval()

    for batch in tqdm(test_loader, total=len(test_loader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        
        ##INPUTS:
        syn_input_ids = batch['synopsis']['input_ids'].type(torch.long).to(device_)
        syn_attention_mask = batch['synopsis']['attention_mask'].type(torch.long).to(device_)

        char_input_ids = batch['descriptions']['input_ids'].type(torch.long).to(device_)
        char_attention_mask = batch['descriptions']['attention_mask'].type(torch.long).to(device_)

        img_key = 'pixel_values' #'input_ids'
        img_input_ids = batch['portraits'][img_key].type(torch.float).to(device_)


        with torch.no_grad(): # Telling the model not to compute or store gradients, saving memory and speeding up validation
            outputs = model(syn_input_ids=syn_input_ids, char_input_ids=char_input_ids, syn_attention_mask=syn_attention_mask, char_attention_mask=char_attention_mask, img_input_ids=img_input_ids).to(device_)
            logits = outputs

            predictions += logits.squeeze().detach().cpu().tolist()
            predictions_loss = torch.Tensor(logits.squeeze().detach().cpu()).to(device_)

            loss = loss_fn(predictions_loss, torch.Tensor(batch['labels'].float()).to(device_))
        #loss.backward()

            total_loss += loss.item()

    avg_epoch_loss = total_loss / len(test_loader)

    return true_labels, predictions, avg_epoch_loss





from Ca_Naxca import regression_report
from transformers import get_linear_schedule_with_warmup

##TRAIN THE MODEL.

optimizer_ = torch.optim.AdamW(model.parameters(), lr = learning_rate, eps = eps_value)
total_steps = len(train_dataloader) * epochs
scheduler_ = get_linear_schedule_with_warmup(optimizer_, num_warmup_steps = 0, num_training_steps = total_steps)
loss_fn = nn.MSELoss()


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
    #reporte.plot(f".//final_reportes//predVSreal_{epoch+1}.png")
    #reps = reporte.display()
    #print(reps)
    #reps.to_csv(".//final_reportes//reporte_{epoch+1}.csv")

    for x in [[valid_labels[i], valid_predict[i]] for i in range(10)]:
        print(x)

    print("  train_loss: %.5f - val_loss: %.5f "%(train_loss, val_loss))
    print()
    trainval.append([train_loss, val_loss])


#import json

#jsonfile = open(".//final_reportes//train_val_loss.json", "w")
#json.dump(trainval, jsonfile)
#jsonfile.close()


def reverse_scaler(values_base, ratio, values_target):
    minim = min(values_base)
    maxim = max(values_base)
    return [minim + t*(maxim - minim)/ratio for t in values_target]

los_labels = anime_test["Score"].values.tolist()

predicciones = reverse_scaler(los_labels, 1, valid_predict)
anime_test["preds"] = predicciones
anime_test.to_csv(".//preds//preds_all.csv")
