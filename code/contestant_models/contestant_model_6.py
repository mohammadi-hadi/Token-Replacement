import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer
import torch
import torch.nn as nn
import psutil
from transformers import (BertTokenizer, BertForSequenceClassification,
                          XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
                          DistilBertTokenizer, DistilBertForSequenceClassification)
from torch.utils.data import DataLoader

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERTModel, self).__init__()

        # Load and move to the correct device directly here
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels).to(device)
        
        self.xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.xlm_roberta_model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels).to(device)
        
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=num_labels).to(device)

    def forward(self, input_ids):
        # Return stacked logits
        bert_logits = self.bert_model(input_ids).logits
        xlm_roberta_logits = self.xlm_roberta_model(input_ids).logits
        distilbert_logits = self.distilbert_model(input_ids).logits

        all_logits = torch.stack((bert_logits, xlm_roberta_logits, distilbert_logits), dim=0)
        return all_logits

GRADIENT_ACCUMULATION_STEPS = 2

def evaluate_model(model, loader, criterion):
    model.eval()
    true_labels = []
    predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(inputs)
            final_logits, _ = torch.mode(logits, dim=0)  # Voting mechanism
            loss = criterion(final_logits, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(final_logits, dim=1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    f1 = f1_score(true_labels, predictions, average='macro')
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')

    return average_loss, f1, acc, precision, recall, true_labels, predictions

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def load_model():
    model_path = "class_best_model.pth"
    model = CustomBERTModel(num_labels=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_dataframe(df, model, text_column, label_column):
    predictions = []
    true_labels = df[label_column].tolist()

    for index, row in df.iterrows():
        text = row[text_column]
        
        # Tokenize the text
        encoded_text = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
        input_ids = encoded_text['input_ids'].to(device)

        # Predict
        with torch.no_grad():
            logits = model(input_ids)
            prediction = torch.argmax(logits, dim=1)
            predictions.append(prediction.item())
    
    return true_labels, predictions

# Compute metrics
def compute_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, f1


