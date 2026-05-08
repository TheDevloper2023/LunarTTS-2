import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel


device = "cuda" if torch.cuda.is_available() else "cpu"

class TextStyleGen(nn.Module):
    def __init__(self, bert_dim, enc_dim, model_name="cardiffnlp/twitter-roberta-base-emoji-latest" ,dropout=0.1):
        super().__init__()
        # Load RoBERT

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


        self.attn = nn.Linear(bert_dim, 1)                    # simple attention score
        
        self.proj = nn.Sequential(
            nn.Linear(bert_dim * 2, bert_dim),             # concat → enc_dim
            nn.LayerNorm(bert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    

    def forward(self, text, target_seq_len, bert_train=False):
        # RoBERTa
        inputs = self.tokenizer(text, return_tensors="pt",padding=True,truncation=True)
        inputs = inputs.to(device)
        if not bert_train:
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)


        hidden = outputs.last_hidden_state
        pooler = outputs.last_hidden_state[:, 0]
        
        weights = torch.softmax(self.attn(hidden), dim=1)
        bert_attn = (weights * hidden).sum(dim=1)

        combined = torch.cat([bert_attn, pooler], dim=-1)

        bert_embedding = self.proj(combined)
        bert_embedding = bert_embedding.unsqueeze(1).expand(-1, target_seq_len, -1)


        return bert_embedding



    