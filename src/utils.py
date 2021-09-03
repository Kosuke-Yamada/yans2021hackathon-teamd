import os
import time
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def decode_attr_bio(totals, idx2attr, idx2bio):
    bios = []
    for attr_idx, attr in idx2attr.items():
        bios.append([idx2bio[bio_idx]+'-'+attr if idx2bio[bio_idx] != 'O' else 'O' for bio_idx in totals[attr_idx]])
    return bios              

def operate_bio(pre_bio, bio):
    ope = {'insert':False, 'start':False, 'end':False}
    if pre_bio == 'B':        
        if bio == 'B':
            ope['insert'] = True            
            ope['start'] = True
            ope['end'] = True
        elif bio == 'I':
            ope['end'] = True
        else:          
            ope['insert'] = True        
    elif pre_bio == 'I':
        if bio == 'B':
            ope['insert'] = True     
            ope['start'] = True
            ope['end'] = True
        elif bio == 'I':
            ope['end'] = True        
        else:       
            ope['insert'] = True    
    else:
        if bio == 'B':
            ope['start'] = True
            ope['end'] = True
        elif bio == 'I':
            ope['start'] = True
            ope['end'] = True
        else:            
            pass
    return ope  

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs