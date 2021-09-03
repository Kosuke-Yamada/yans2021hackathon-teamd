import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, attention_masks, labels = list(zip(*batch))
    inputs = pad_sequence(inputs, batch_first=True)
    attention_masks = pad_sequence(attention_masks, padding_value=False, batch_first=True)
    labels = pad_sequence([l.transpose(0, 1).reshape(-1) for l in labels], batch_first=True, \
                          padding_value=-1).reshape(len(labels), -1, labels[0].shape[0]).transpose(1, 2).long()
    return inputs, attention_masks, labels

def make_title(plain):
    for line in plain:
        pos = line.find(' - Wikipedia Dump')
        if pos != -1:
            title = line[:pos]
            break
    return title                
    
class SearchBottomLine:
    def __init__(self):
        self.bottom_flag = False
    def search(self, line):
        if self.bottom_flag == False:
            if re.search(r'」から取得', line) is not None:
                self.bottom_flag = True
            if line in ['脚注\n', '参考文献\n', '外部リンク\n', '関連項目\n']:
                self.bottom_flag = True
        if self.bottom_flag == True:
            return True
        return False
    
def remove_char(char_dict, line_flag):
    if re.match(r'\S', char_dict['char']) is None:
        return True
    if char_dict['line_id'] <= 26:
        return True  
    if line_flag == True:
        return True
    return False

def remove_line(line):
    if line[0] == '^':
        line = ' '*len(line)
    pattern = r'「?(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|][-A-Za-z0-9+&@#/%=~_|]+[-A-Za-z0-9+&@#/%=~_|]*?'
    if re.search(pattern, line):
        line = re.sub(pattern, ' '*len(re.search(pattern,line).group()), line) 
    pattern = r'((http|ftp|https)://)*(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?'
    if re.search(pattern, line):
        line = re.sub(pattern, ' '*len(re.search(pattern,line).group()), line) 
    pattern = r'ファイル:(.+?\..{3})'
    if re.search(pattern, line):
        line = re.sub(pattern, ' '*len(re.search(pattern, line).group()), line)
    pattern = r'移動先:					案内、					検索'
    if re.search(pattern, line):
        line = re.sub(pattern, ' '*len(re.search(pattern, line).group()), line)
    pattern = r'出典: フリー百科事典『ウィキペディア（Wikipedia）』'
    if re.search(pattern, line):
        line = re.sub(pattern, ' '*len(re.search(pattern, line).group()), line)        
    return line

class CharbertDataset(Dataset):
    def __init__(self, page2plain, tokenizer, attr2idx, bio2idx, max_length, block, df_anno=None):
                
        self.page2plain = page2plain
        self.tokenizer = tokenizer
        self.attr2idx = attr2idx
        self.bio2idx = bio2idx
        self.block = block
        
        self.df_anno = df_anno
        self.max_length = max_length        
        
        self._make_dataset()            

    def __len__(self):            
        return self.data_num
    
    def __getitem__(self, idx):        
        return self.out_inputs[idx], self.out_attention_masks[idx], self.out_labels[idx]
        
    def _make_loa2bio(self, page_id):
        loa2bio = {}
        if self.df_anno is not None:
            for df_dict in self.df_anno[self.df_anno['page_id'] == page_id].to_dict('records'):    
                sl = df_dict['text_offset']['start']['line_id']
                so = df_dict['text_offset']['start']['offset']
                el = df_dict['text_offset']['end']['line_id']
                eo = df_dict['text_offset']['end']['offset']        
                attr = df_dict['attribute']
                for il, l in enumerate(range(sl, el+1)):
                    if l not in loa2bio:
                        loa2bio[l] = {}
                    for io, o in enumerate(range(so, eo)):
                        if o not in loa2bio[l]:
                            loa2bio[l][o] = {}                    
                        if il == 0 and io == 0:
                            loa2bio[l][o][attr] = 'B'
                        else:
                            loa2bio[l][o][attr] = 'I'
        return loa2bio
                        
    def _make_char_list_line(self, plain):
        sbl = SearchBottomLine()
        char_idx = 0
        new_char_list, new_char_idx = [], 0
        block_id, block_idx = 0, 0
        for line_id, line in enumerate(plain):
            line_flag = False
            line = remove_line(line)
            line_flag = sbl.search(line)
            for offset, char in enumerate(line):                
                char_dict = {'line_id':line_id, 'offset':offset, \
                             'char_idx':char_idx, 'char':char}
                char_idx += 1 
                
                if remove_char(char_dict, line_flag) == True:
                    continue
                    
                if block_idx >= self.max_length-2:
                    continue
                    
                new_char_dict = char_dict.copy()
                new_char_dict['new_char_idx'] = new_char_idx
                new_char_dict['block_id'] = block_id
                new_char_dict['block_idx'] = block_idx
                new_char_list.append(new_char_dict)
                new_char_idx += 1  
                block_idx += 1
            block_id += 1
            block_idx = 0
        return new_char_list
    
    def _make_char_list_char(self, plain):
        sbl = SearchBottomLine()
        char_idx = 0
        new_char_list, new_char_idx = [], 0
        for line_id, line in enumerate(plain):
            line_flag = False
            line = remove_line(line)
            line_flag = sbl.search(line)
            for offset, char in enumerate(line):                
                char_dict = {'line_id':line_id, 'offset':offset, \
                             'char_idx':char_idx, 'char':char}                            
                char_idx += 1 
                
                if remove_char(char_dict, line_flag) == True:
                    continue
                    
                new_char_dict = char_dict.copy()
                new_char_dict['new_char_idx'] = new_char_idx
                new_char_dict['block_id'] = new_char_idx // (self.max_length-2)
                new_char_dict['block_idx'] = new_char_idx % (self.max_length-2)                
                new_char_list.append(new_char_dict)
                new_char_idx += 1  
        return new_char_list
    
    def _make_io(self, df_block, loa2bio):        
        chars = ['[CLS]']
        labels = -torch.ones(len(self.attr2idx), len(df_block)+2).long()
        for char_dict in df_block.to_dict('records'):
            chars.append(char_dict['char'])
            
            labels[:, char_dict['block_idx']+1] = self.bio2idx['O']
            if char_dict['line_id'] in loa2bio:
                if char_dict['offset'] in loa2bio[char_dict['line_id']]:
                    attr2bio = loa2bio[char_dict['line_id']][char_dict['offset']]
                    for attr, bio in attr2bio.items():
                        labels[self.attr2idx[attr], char_dict['block_idx']+1] = self.bio2idx[bio]
                    
        chars += ['[SEP]']
        input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(chars))
        attention_masks = torch.ones(input_ids.shape).bool()
        
        return input_ids, attention_masks, labels
        
    def _make_dataset(self, ):
        self.out_inputs, self.out_attention_masks = [], []
        self.out_labels = []
        self.data_num = 0
        self.df_new = {}
        self.page2title = {}
        for page_id in sorted(self.page2plain.keys()):
            loa2bio = self._make_loa2bio(page_id)
            
            plain = self.page2plain[page_id]
            self.page2title[page_id] = make_title(plain)
            if self.block == 'line':
                new_char_list = self._make_char_list_line(plain)
            else:
                new_char_list = self._make_char_list_char(plain)
            
            df_new = pd.DataFrame(new_char_list)
            self.df_new[page_id] = df_new
            for block_id in sorted(set(df_new['block_id'])):    
                df_block = df_new[df_new['block_id'] == block_id]
                inputs, attention_masks, labels = self._make_io(df_block, loa2bio)
                
                self.out_inputs.append(inputs)
                self.out_attention_masks.append(attention_masks)
                self.out_labels.append(labels)                
                self.data_num += 1                

class ShibaDataset(Dataset):
    def __init__(self, page2plain, tokenizer, attr2idx, bio2idx, max_length, block, df_anno=None):
                
        self.page2plain = page2plain
        self.attr2idx = attr2idx
        self.bio2idx = bio2idx
        
        self.df_anno = df_anno
        self.max_length = max_length
        self.block = block
        
        self.tokenizer = tokenizer
        self.cls = self.tokenizer.CLS
        self.pad = self.tokenizer.PAD    
        
        self._make_dataset()            

    def __len__(self):            
        return self.data_num
    
    def __getitem__(self, idx):        
        return self.out_inputs[idx], self.out_attention_masks[idx], self.out_labels[idx]
    
    def _make_loa2bio(self, page_id):
        loa2bio = {}
        if self.df_anno is not None:
            for df_dict in self.df_anno[self.df_anno['page_id'] == page_id].to_dict('records'):    
                sl = df_dict['text_offset']['start']['line_id']
                so = df_dict['text_offset']['start']['offset']
                el = df_dict['text_offset']['end']['line_id']
                eo = df_dict['text_offset']['end']['offset']        
                attr = df_dict['attribute']
                for il, l in enumerate(range(sl, el+1)):
                    if l not in loa2bio:
                        loa2bio[l] = {}
                    for io, o in enumerate(range(so, eo)):
                        if o not in loa2bio[l]:
                            loa2bio[l][o] = {}                    
                        if il == 0 and io == 0:
                            loa2bio[l][o][attr] = 'B'
                        else:
                            loa2bio[l][o][attr] = 'I'
        return loa2bio
    
    def _make_char_list_line(self, plain):
        sbl = SearchBottomLine()
        char_idx = 0
        new_char_list, new_char_idx = [], 0
        block_id, block_idx = 0, 0
        for line_id, line in enumerate(plain):
            line_flag = False
            line = remove_line(line)
            line_flag = sbl.search(line)
            for offset, char in enumerate(line):                
                char_dict = {'line_id':line_id, 'offset':offset, \
                             'char_idx':char_idx, 'char':char}
                char_idx += 1 
                
                if remove_char(char_dict, line_flag) == True:
                    continue
                    
                if block_idx >= self.max_length-2:
                    continue
                    
                new_char_dict = char_dict.copy()
                new_char_dict['new_char_idx'] = new_char_idx
                new_char_dict['block_id'] = block_id
                new_char_dict['block_idx'] = block_idx
                new_char_list.append(new_char_dict)
                new_char_idx += 1  
                block_idx += 1
            block_id += 1
            block_idx = 0
        return new_char_list
                        
    def _make_char_list_char(self, plain):
        sbl = SearchBottomLine()
        char_idx = 0
        new_char_list, new_char_idx = [], 0
        for line_id, line in enumerate(plain):
            line_flag = False
            line = remove_line(line)
            line_flag = sbl.search(line)
            for offset, char in enumerate(line):                
                char_dict = {'line_id':line_id, 'offset':offset, \
                             'char_idx':char_idx, 'char':char}
                char_idx += 1 
                
                if remove_char(char_dict, line_flag) == True:
                    continue
                    
                new_char_dict = char_dict.copy()
                new_char_dict['new_char_idx'] = new_char_idx
                new_char_dict['block_id'] = new_char_idx // (self.max_length-1)
                new_char_dict['block_idx'] = new_char_idx % (self.max_length-1)
                new_char_list.append(new_char_dict)
                new_char_idx += 1  
        return new_char_list
    
    def _make_io(self, df_block, loa2bio):        
        chars = [self.cls]        
        if len(df_block) <= 3:            
            labels = -torch.ones(len(self.attr2idx), len(df_block)+4).long()
        else:
            labels = -torch.ones(len(self.attr2idx), len(df_block)+1).long()
        for char_dict in df_block.to_dict('records'):
            chars.append(ord(char_dict['char']))                        
            
            labels[:, char_dict['block_idx']+1] = self.bio2idx['O']
            if char_dict['line_id'] in loa2bio:
                if char_dict['offset'] in loa2bio[char_dict['line_id']]:
                    attr2bio = loa2bio[char_dict['line_id']][char_dict['offset']]
                    for attr, bio in attr2bio.items():
                        labels[self.attr2idx[attr], char_dict['block_idx']+1] = self.bio2idx[bio]
        if len(df_block) <= 3:
            chars += [self.pad, self.pad, self.pad]
        
        input_ids = torch.LongTensor(chars)
        attention_masks = (input_ids == -1)
        
        return input_ids, attention_masks, labels
        
    def _make_dataset(self, ):
        self.out_inputs, self.out_attention_masks = [], []
        self.out_labels = []
        self.data_num = 0
        self.df_new = {}
        self.page2title = {}
        for page_id in sorted(self.page2plain.keys()):
            loa2bio = self._make_loa2bio(page_id)
            
            plain = self.page2plain[page_id]
            self.page2title[page_id] = make_title(plain)
            if self.block == 'line':
                new_char_list = self._make_char_list_line(plain)
            else:
                new_char_list = self._make_char_list_char(plain)
            df_new = pd.DataFrame(new_char_list)
            self.df_new[page_id] = df_new
            for block_id in sorted(set(df_new['block_id'])):    
                df_block = df_new[df_new['block_id'] == block_id]
                inputs, attention_masks, labels = self._make_io(df_block, loa2bio)
                
                self.out_inputs.append(inputs)
                self.out_attention_masks.append(attention_masks)
                self.out_labels.append(labels)                
                self.data_num += 1