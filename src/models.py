import torch
import torch.nn as nn
                
class CharbertForSequenceLabeling(nn.Module):
    def __init__(self, bert, attr_size, label_size):
        super(CharbertForSequenceLabeling, self).__init__()
        self.attr_size = attr_size
        self.label_size = label_size
        
        self.bert = bert
        
        self.config = self.bert.config
        self.config.attr_size = self.attr_size
        self.config.label_size = self.label_size
        hidden_size = self.bert.config.hidden_size
        
        linear_layers = [torch.nn.Linear(hidden_size, hidden_size) for _ in range(attr_size)]
        self.linear_layers = nn.ModuleList(linear_layers)
        self.relu = nn.ReLU()
        
        label_layers = [torch.nn.Linear(hidden_size, self.label_size) for _ in range(attr_size)]
        self.label_layers = nn.ModuleList(label_layers)
        
        self.dropout = torch.nn.Dropout(p=self.bert.config.hidden_dropout_prob)        

        self.softmax = torch.nn.Softmax(dim=-1)        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)        

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask)[0]        
        sequence_outputs = self.dropout(outputs)
        hiddens = [self.relu(layer(sequence_outputs)) for layer in self.linear_layers]
        logits = [layer(hiddens) for layer, hiddens in zip(self.label_layers, hiddens)]
        probs = [self.softmax(logit) for logit in logits]                      
            
        loss = None
        if labels is not None:                
            loss = 0
            for logit, label in zip(logits, labels.transpose(0, 1)):
                loss += self.criterion(logit.reshape(-1, self.label_size), label.reshape(-1)) / len(label)        
        return loss, probs
                
class ShibaForSequenceLabeling(nn.Module):
    def __init__(self, shiba, attr_size, label_size):
        super(ShibaForSequenceLabeling, self).__init__()
        self.attr_size = attr_size
        self.label_size = label_size
        
        self.shiba = shiba
        
        self.config = self.shiba.config
        self.config.attr_size = self.attr_size
        self.config.label_size = self.label_size
        hidden_size = self.shiba.config.hidden_size
        
        linear_layers = [torch.nn.Linear(hidden_size, hidden_size) for _ in range(attr_size)]
        self.linear_layers = nn.ModuleList(linear_layers)
        self.relu = nn.ReLU()
        
        label_layers = [torch.nn.Linear(hidden_size, self.label_size) for _ in range(attr_size)]
        self.label_layers = nn.ModuleList(label_layers)
        
        self.dropout = torch.nn.Dropout(p=self.shiba.config.dropout)        

        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)        

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.shiba(input_ids, attention_mask)['embeddings']
        sequence_outputs = self.dropout(outputs)
        hiddens = [self.relu(layer(sequence_outputs)) for layer in self.linear_layers]
        logits = [layer(hiddens) for layer, hiddens in zip(self.label_layers, hiddens)]
        probs = [self.softmax(logit) for logit in logits]                      
            
        loss = None
        if labels is not None:                
            loss = 0
            for logit, label in zip(logits, labels.transpose(0, 1)):
                loss += self.criterion(logit.reshape(-1, self.label_size), label.reshape(-1)) / len(label)        
        return loss, probs