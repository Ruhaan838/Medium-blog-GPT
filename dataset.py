import torch
from torch.utils.data import DataLoader, Dataset, random_split

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
import pandas as pd

import subprocess
import kagglehub

def get_text(dataset):
    for i in range(len(dataset)):
        yield str(dataset['text'][i]) + " " + str(dataset['title'][i])

def get_tokenizer(dataset, file_name):
    path = os.getcwd() + file_name
    if not os.path.exists(path):
        tokenizer = Tokenizer(WordLevel(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<|unk|>", "<|pad|>", "<|sos|>", "<|eos|>"])
        tokenizer.train_from_iterator(get_text(dataset), trainer=trainer)
        tokenizer.save(path)
    else:
        tokenizer = Tokenizer.from_file(path)
    return tokenizer

class MediumDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len) -> None:

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        self.sos = torch.tensor([tokenizer.token_to_id("<|sos|>")], dtype=torch.int64)
        self.eos = torch.tensor([tokenizer.token_to_id("<|eos|>")], dtype=torch.int64)
        self.pad = torch.tensor([tokenizer.token_to_id("<|pad|>")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, indx):
        title = self.dataset.iloc[indx]['title']
        text = self.dataset.iloc[indx]['text']
        
        title_encode = self.tokenizer.encode(title).ids
        text_encode = self.tokenizer.encode(text).ids
        
        pad_title = self.seq_len - len(title_encode) - 1
        pad_label = self.seq_len - len(text_encode) - 1
        

        input_tensor = torch.cat((
            self.sos,
            torch.tensor(title_encode, dtype=torch.int64),
            torch.tensor([self.pad] * pad_title, dtype=torch.int64),
        ))

        label = torch.cat((
            torch.tensor(text_encode, dtype=torch.int64),
            torch.tensor([self.pad] * pad_label, dtype=torch.int64),
            self.eos
        ))
        
        return {
            "decoder_input":input_tensor,
            "label":label,
            "title":title,
            "text":text
        }
        
def get_dataloder(seq_len:int, batch_size:int, workers:int=0):
    
    if os.path.exists("2/medium_articles.csv"):
        data = pd.read_csv("2/medium_articles.csv")
    else:    
        path = kagglehub.dataset_download("fabiochiusano/medium-articles")
        subprocess.call(["mv", path, "."])
        data = pd.read_csv("2/medium_articles.csv")
    
    max_len = 0
    
    tokenizer = get_tokenizer(data, "tokenizer_text.json")
    
    for i in range(len(data)):
        ids = tokenizer.encode(str(data['text'][i]) + " " + str(data['title'][i])).ids
        max_len = max(max_len, len(ids))
    
    print(f"Max Seq_len need to set {max_len}")
    
    full_dataset = MediumDataset(data, tokenizer, seq_len)
    
    train_size = int(len(full_dataset) * 0.8)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True, num_workers=workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, 1, shuffle=True, num_workers=workers, pin_memory=True)
    
    
    return train_dataloader, val_dataloader, tokenizer

if __name__ == "__main__": 
    train_dataloader, val_dataloader, tokenizer = get_dataloder(25000,8)
    count = 0
    for batch in train_dataloader:
        if count % 1000 == 0:
            print(batch['decoder_input'])
            print(batch['label'])
        count += 1
        