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
        self.l3 = torch.nn.Linear(bert_dim, output_dim)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class BertLinear(nn.Module):
    """Linear layer after a BERT layer."""

    def __init__(self, bert_type: str, frozen: bool, linear_dim: int, name: str = None) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.frozen = frozen
        self.linear_dim = linear_dim
        self.name = name or f"{'frozen_' if frozen else ''}{bert_type}_linear_{linear_dim}"
        if bert_type == "bert":
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        if frozen:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_dim = self.bert.config.hidden_size
        if linear_dim > 0:
            self.linear = nn.Linear(self.bert_dim, linear_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled_output = self.bert(
            input_ids=input_ids, attention_mask=(input_ids != 0).long(), return_dict=True
        )[
            "pooler_output"
        ]  # (batch_size, nb_int, bert_dim)

        if self.linear_dim > 0:
            pooled_output = self.linear(pooled_output)

        return pooled_output