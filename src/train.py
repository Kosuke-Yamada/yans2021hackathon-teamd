#-*-coding:utf-8-*-

import os
import re
import json
import time
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
from shiba import Shiba, CodepointTokenizer, get_pretrained_state_dict

from datasets import CharbertDataset, ShibaDataset, collate_fn
from models import CharbertForSequenceLabeling, ShibaForSequenceLabeling
from utils import epoch_time, decode_attr_bio, set_seed

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_plain_path", type=str)
    parser.add_argument("--input_annotation_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--block", type=str)        
    parser.add_argument("--model", type=str)        
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--max_epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)            
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cuda", type=int)
    return parser.parse_args()

if __name__ == "__main__":   
    args = parse_arg()
    
    INPUT_PLAIN_PATH = args.input_plain_path
    INPUT_ANNOTATION_PATH = args.input_annotation_path
    OUTPUT_PATH = args.output_path
    CATEGORY = args.category
    BLOCK = args.block
    MODEL = args.model         
    MAX_LENGTH = args.max_length
    MAX_EPOCH = args.max_epoch
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate        
    GRAD_CLIP = args.grad_clip    
    SEED = args.seed
    CUDA = args.cuda   
    
    OUTPUT_PATH = OUTPUT_PATH+CATEGORY.lower()+'_'+MODEL.lower()+'_'+BLOCK.lower()+'/'
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    params = {
            'seed':SEED,
            'batch_size':BATCH_SIZE,
            'learing_rate':LEARNING_RATE,        
            'max_length':MAX_LENGTH,
            'max_epoch':MAX_EPOCH,
            'grad_clip':GRAD_CLIP
            }
    with open(OUTPUT_PATH+'params.json', 'w') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)            
        
    set_seed(SEED)
    device = torch.device("cuda:"+str(CUDA) if torch.cuda.is_available() else "cpu")
    
    df = pd.read_json(INPUT_ANNOTATION_PATH+CATEGORY+'_dist.json', orient='records', lines=True)
    
    attr2idx = {attr:i for i, attr in enumerate(sorted(set(df['attribute'])))}
    idx2attr = {v:k for k, v in attr2idx.items()}
    bio2idx = {'B':0, 'I':1, 'O':2}
    idx2bio = {v:k for k, v in bio2idx.items()}
    
    page_id_list = sorted(set(df['page_id']))
    random.shuffle(page_id_list)
    page2tvt = {}
    page2tvt.update({i:'train' for i in page_id_list[:900]})
    page2tvt.update({i:'valid' for i in page_id_list[900:950]})
    page2tvt.update({i:'test' for i in page_id_list[950:]})
    df['tvt'] = df['page_id'].map(page2tvt)
    
    df_train = df[df['tvt'] == 'train']
    page_id_list = sorted(set(df_train['page_id']))
    train_page2plain = {}
    for page_id in page_id_list:  
        with open(INPUT_PLAIN_PATH+CATEGORY+'/'+str(page_id)+'.txt', 'r') as f:
            train_page2plain[page_id] = f.readlines() 
            
    df_valid = df[df['tvt'].isin(['valid', 'test'])]
    page_id_list = sorted(set(df_valid['page_id']))
    valid_page2plain = {}
    for page_id in page_id_list:  
        with open(INPUT_PLAIN_PATH+CATEGORY+'/'+str(page_id)+'.txt', 'r') as f:
            valid_page2plain[page_id] = f.readlines() 
    
    if MODEL == 'charbert':
        pretrained_model = 'cl-tohoku/bert-base-japanese-char-whole-word-masking'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        bert = AutoModel.from_pretrained(pretrained_model)        
        train_ds = CharbertDataset(train_page2plain, tokenizer, attr2idx, bio2idx, MAX_LENGTH, BLOCK, df_train)
        model = CharbertForSequenceLabeling(bert, attr_size=len(attr2idx), label_size=len(bio2idx))
    else:
        tokenizer = CodepointTokenizer()
        shiba = Shiba()
        shiba.load_state_dict(get_pretrained_state_dict())                
        train_ds = ShibaDataset(train_page2plain, tokenizer, attr2idx, bio2idx, MAX_LENGTH, BLOCK, df_train)        
        model = ShibaForSequenceLabeling(shiba, attr_size=len(attr2idx), label_size=len(bio2idx))
        
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    log_dict = {}
    best_valid_f1, best_epoch, early_stopping = 0, 0, 0
    for epoch in range(MAX_EPOCH):  
        print()
        print(f'Epoch: {epoch:02}')
        
        start_time = time.time()
        model.to(device).train()        
        train_loss = 0
        for inputs, am, labels in train_dl:
            output = model(inputs.to(device), am.to(device), labels.to(device))
            loss = output[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()        
            optimizer.zero_grad()
            train_loss += loss.item()  
        train_loss /= len(train_dl)
        end_time = time.time()
        train_mins, train_secs = epoch_time(start_time, end_time)
        print(f'Time: {train_mins}m {train_secs}s')        
        print(f'Train Loss: {train_loss:.9f}')          
        
        start_time = time.time()
        valid_f1 = 0
        for page_id in list(valid_page2plain.keys()):
            page2plain = {page_id:valid_page2plain[page_id]}
            if MODEL == 'charbert':
                ds = CharbertDataset(page2plain, tokenizer, attr2idx, bio2idx, MAX_LENGTH, BLOCK, df_valid)
                dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
            else:
                ds = ShibaDataset(page2plain, tokenizer, attr2idx, bio2idx, MAX_LENGTH, BLOCK, df_valid)
                dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

            _total_labels, _total_preds = torch.LongTensor(), torch.LongTensor()
            for inputs, am, labels in dl:
                with torch.no_grad():            
                    model.to(device).eval()
                    output = model(inputs.to(device), am.to(device), labels.to(device))        
                probs = torch.stack(output[1]).transpose(0, 1).cpu()
                preds = probs.argmax(axis=-1)

                _total_labels = torch.cat([_total_labels, labels.transpose(0, 1).reshape(labels.shape[1], -1)], axis=1)
                _total_preds = torch.cat([_total_preds, preds.transpose(0, 1).reshape(preds.shape[1], -1)], axis=1)

            total_labels = _total_labels[(_total_labels != -1).nonzero(as_tuple=True)].reshape(_total_labels.shape[0], -1)
            total_preds = _total_preds[(_total_labels != -1).nonzero(as_tuple=True)].reshape(_total_preds.shape[0], -1)

            bio_labels = decode_attr_bio(total_labels.tolist(), idx2attr, idx2bio)
            bio_preds = decode_attr_bio(total_preds.tolist(), idx2attr, idx2bio)            
            valid_f1 += f1_score(bio_labels, bio_preds)
        valid_f1 /= len(valid_page2plain.keys())        
        end_time = time.time()
        valid_mins, valid_secs = epoch_time(start_time, end_time)
        print(f'Time: {valid_mins}m {valid_secs}s')        
        print(f'Valid F1: {valid_f1:.6f}')
        
        torch.save(model.to('cpu').state_dict(), OUTPUT_PATH+'model_'+str(epoch).zfill(2)+'.pt')

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.to('cpu').state_dict(), OUTPUT_PATH+'best_model.pt')
            best_epoch = epoch        
            early_stopping = 0
            print('***BEST_VALID_F1***')
        else:
            early_stopping += 1

        log_dict[str(epoch)] = {
                                'epoch':epoch,
                                'train_epoch_mins':train_mins,
                                'train_epoch_secs':train_secs,
                                'valid_epoch_mins':valid_mins,
                                'valid_epoch_secs':valid_secs,
                                'train_loss':train_loss,
                                'valid_f1':valid_f1,
                                'best_epoch':best_epoch,
                                'best_valid_f1': best_valid_f1,                            
                                }
        with open(OUTPUT_PATH+'log.json', 'w') as f:
            json.dump(log_dict, f, indent=4, ensure_ascii=False)

        if early_stopping >= 10:
            break