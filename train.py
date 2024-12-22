from models import GPT2
from config import Config
from dataset import get_dataloder

from tqdm import tqdm
import os

from torch import nn
from torch import optim

def get_model(tokenizer):
    
    model = GPT2(Config.N, tokenizer.get_vocab_size(), 
                 Config.D_MODEL, Config.D_FF, 
                 Config.HEAD_SIZE, Config.SEQ_LEN)
    
    model = model.to(Config.DEVICE)
    return model

def validation(model, tokenizer, ):
    pass

def train(EPOCH):
    
    os.makedirs(Config.RESULT_PATH, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_dataloder(Config.SEQ_LEN, Config.BATCH_SIZE, Config.WORKERS)
    model = get_model(tokenizer)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<|pad|>"), label_smoothing=0.1).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr = Config.LR, eps=1e-9)
    
    for epoch in range(EPOCH):
        model.train()
        for batch in (pbar:=tqdm(train_dataloader, desc=f"Training | Epoch {epoch+1}/{EPOCH}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
            
            inputs = batch['decoder_input'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            predict = model.decode(inputs)
            out = model.projection(predict)
            
            loss = loss_fn(out.view(-1, tokenizer.get_vocab_size()), labels.view(-1))
            pbar.set_postfix(Loss = loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            
            
if __name__ == "__main__":
    train(1)
            
