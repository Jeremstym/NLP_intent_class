import json
from typing import Dict, List

import torch
from torch import nn
from transformers import BertModel


class BERTClass(torch.nn.Module):
    def __init__(self, dropout=0.3, bert_dim=768, output_dim=4, frozen=True):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        if frozen:
            for param in self.l1.parameters():
                param.requires_grad = False
        self.l2 = torch.nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.l3 = torch.nn.Linear(bert_dim, output_dim)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output_2 = self.relu(output_2)
        output = self.l3(output_2)
        return output
