
import json

def get_hyperparams():
    with open("best_params.json", "r") as f:
        hyperparams = json.load(f)
    return hyperparams


import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, XLMRobertaModel, DistilBertModel
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERTModel, self).__init__()

        # BERT model
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

        # XLM-Roberta model
        self.xlm_roberta_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

        # DistilBERT model
        self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Linear layer
        self.fc = nn.Linear(768 * 3, num_labels)

    def forward(self, input_ids):
        bert_output = self.bert_model(input_ids).last_hidden_state
        xlm_roberta_output = self.xlm_roberta_model(input_ids).last_hidden_state
        distilbert_output = self.distilbert_model(input_ids).last_hidden_state

        concatenated = torch.cat((bert_output, xlm_roberta_output, distilbert_output), dim=2)
        concatenated = self.dropout(concatenated)

        out = self.fc(concatenated[:, 0, :])
        return out

# Model Loading function
def load_model():
    model = CustomBERTModel(num_labels=2).to(device)
    # Assuming the model weights are stored in 'best_model.pth'
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    return model

# Prediction function
def predict(text_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    encodings = tokenizer(text_list, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    
    model = load_model()
    
    with torch.no_grad():
        outputs = model(input_ids)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()

    return predictions

