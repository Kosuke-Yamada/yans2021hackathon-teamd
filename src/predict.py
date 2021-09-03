#-*-coding:utf-8-*-

import os
import re
import json
import time
import glob
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
from shiba import Shiba, CodepointTokenizer, get_pretrained_state_dict

from datasets import CharbertDataset, ShibaDataset, collate_fn
from models import CharbertForSequenceLabeling, ShibaForSequenceLabeling
from utils import epoch_time, decode_attr_bio, operate_bio, set_seed

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_plain_path", type=str)
    parser.add_argument("--input_annotation_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--block", type=str)        
    parser.add_argument("--model", type=str)        
    parser.add_argument("--batch_size", type=int)
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
    BATCH_SIZE = args.batch_size
    CUDA = args.cuda   
    
    OUTPUT_PATH = OUTPUT_PATH+CATEGORY.lower()+'_'+MODEL.lower()+'_'+BLOCK.lower()+'/'
    
    with open(OUTPUT_PATH+'params.json', 'r') as f:
        params = dict(json.load(f))        
    SEED = params['seed']
    MAX_LENGTH = params['max_length']
    
    set_seed(SEED)
    device = torch.device("cuda:"+str(CUDA) if torch.cuda.is_available() else "cpu")

    df = pd.read_json(INPUT_ANNOTATION_PATH+CATEGORY+'_dist.json', orient='records', lines=True)
        
    attr2idx = {attr:i for i, attr in enumerate(sorted(set(df['attribute'])))}
    idx2attr = {v:k for k, v in attr2idx.items()}
    bio2idx = {'B':0, 'I':1, 'O':2}
    idx2bio = {v:k for k, v in bio2idx.items()}
            
    page_id_list = [int(path.split('/')[-1][:-4]) for path in sorted(glob.glob(INPUT_PLAIN_PATH+CATEGORY+'/*'))]
    
    pred_page2plain = {}
    for page_id in page_id_list:  
        with open(INPUT_PLAIN_PATH+CATEGORY+'/'+str(page_id)+'.txt', 'r') as f:
            pred_page2plain[page_id] = f.readlines() 
    
    if MODEL == 'charbert':
        pretrained_model = 'cl-tohoku/bert-base-japanese-char-whole-word-masking'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        bert = AutoModel.from_pretrained(pretrained_model)        
        model = CharbertForSequenceLabeling(bert, attr_size=len(attr2idx), label_size=len(bio2idx))
    else:
        tokenizer = CodepointTokenizer()
        shiba = Shiba()
        shiba.load_state_dict(get_pretrained_state_dict())        
        model = ShibaForSequenceLabeling(shiba, attr_size=len(attr2idx), label_size=len(bio2idx))
    model.load_state_dict(torch.load(OUTPUT_PATH+'best_model.pt'))
    
    result_list = []
    for idx, page_id in enumerate(list(pred_page2plain.keys())):    
        print(idx, page_id)
        page2plain = {page_id:pred_page2plain[page_id]}
        if MODEL == 'charbert':
            ds = CharbertDataset(page2plain, tokenizer, attr2idx, bio2idx, MAX_LENGTH, BLOCK, None)            
        else:
            ds = ShibaDataset(page2plain, tokenizer, attr2idx, bio2idx, MAX_LENGTH, BLOCK, None)
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
        total_preds = _total_preds[(_total_labels != -1).nonzero(as_tuple=True)].reshape(_total_preds.shape[0], -1)
        bio_preds = decode_attr_bio(total_preds.tolist(), idx2attr, idx2bio)
        
        new_char_idx_dict = {page_dict['new_char_idx']:page_dict \
                     for page_dict in ds.df_new[page_id].to_dict('records')}        
                
        for attr_idx, bios in enumerate(bio_preds):
            pre_bio = 'O'
            result = {'page_id':page_id, 'title':ds.page2title[page_id], \
                      'attribute':idx2attr[attr_idx], 'text_offset':{}}
            for idx, bio in enumerate(bios):        
                bio = bio.split('-')[0]                    
                ope = operate_bio(pre_bio, bio)
                if ope['insert'] == True:
                    result_list.append(result)
                    result = {'page_id':page_id, 'title':ds.page2title[page_id], \
                      'attribute':idx2attr[attr_idx], 'text_offset':{}}
                if ope['start'] == True:
                    result['text_offset']['start'] = {
                        'line_id': new_char_idx_dict[idx]['line_id'],
                        'offset': new_char_idx_dict[idx]['offset']
                    }
                    result['text_offset']['text'] = new_char_idx_dict[idx]['char']
                if ope['end'] == True:
                    result['text_offset']['end'] = {
                        'line_id': new_char_idx_dict[idx]['line_id'],
                        'offset': new_char_idx_dict[idx]['offset']+1
                    }
                    if ope['start'] == False:
                        result['text_offset']['text'] += new_char_idx_dict[idx]['char']            
                pre_bio = bio
            if bio in ['B', 'I']:
                result_list.append(result)                
        
    df_result = pd.DataFrame(result_list)
    df_result.to_json(OUTPUT_PATH+'predict.json', orient='records', force_ascii=False, lines=True)