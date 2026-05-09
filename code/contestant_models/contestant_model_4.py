import psutil
import torch
import torch.nn as nn
from transformers import (BertTokenizer, BertForSequenceClassification,
                          XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
                          DistilBertTokenizer, DistilBertForSequenceClassification)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward(self, input_ids):
        # Return stacked logits
        bert_logits = self.bert_model(input_ids).logits
        xlm_roberta_logits = self.xlm_roberta_model(input_ids).logits
        distilbert_logits = self.distilbert_model(input_ids).logits

        all_logits = torch.stack((bert_logits, xlm_roberta_logits, distilbert_logits), dim=0)
        return all_logits

# Use this function to load the model
def load_model():
    model = CustomBERTModel(num_labels=2).to(device)
    
    # Load individual models from saved state dicts
    model.bert_model.load_state_dict(torch.load("class_bert_model_weights.pth"))
    model.xlm_roberta_model.load_state_dict(torch.load("class_xlm_roberta_model_weights.pth"))
    model.distilbert_model.load_state_dict(torch.load("class_distilbert_model_weights.pth"))
    
    model.eval()
    return model

def predict(text_list):
    model = load_model()
    
    bert_input = model.bert_tokenizer(text_list, truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].to(device)
    xlmr_input = model.xlm_roberta_tokenizer(text_list, truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].to(device)
    distil_input = model.distilbert_tokenizer(text_list, truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        bert_outputs = model.bert_model(bert_input).logits
        xlmr_outputs = model.xlm_roberta_model(xlmr_input).logits
        distil_outputs = model.distilbert_model(distil_input).logits
    
    # Ensemble logic
    stacked_logits = torch.stack((bert_outputs, xlmr_outputs, distil_outputs), dim=0)
    final_logits, _ = torch.mode(stacked_logits, dim=0)
    
    predictions = torch.argmax(final_logits, dim=1).cpu().numpy().tolist()
    return predictions
