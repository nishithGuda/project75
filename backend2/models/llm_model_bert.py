import torch
import torch.nn as nn
from transformers import BertModel


class LLMQueryElementClassifier(nn.Module):
    def __init__(self, query_hidden_dim=768, element_feature_dim=20, fusion_hidden_dim=128):  # Changed to 20
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(query_hidden_dim + element_feature_dim,
                      fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim//2, 1)
        )

    def forward(self, query_input_ids, query_attention_mask, element_features):
        outputs = self.bert(
            query_input_ids, attention_mask=query_attention_mask)
        query_cls_embedding = outputs.last_hidden_state[:, 0, :]

        x = torch.cat([query_cls_embedding, element_features], dim=1)
        logits = self.classifier(x)
        return logits.squeeze(-1)
