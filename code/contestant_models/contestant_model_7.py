import torch
import torch.nn as nn
from transformers import (BertTokenizer, BertForSequenceClassification,
                          XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
                          DistilBertTokenizer, DistilBertForSequenceClassification)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERTModel, self).__init__()

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)

        self.xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.xlm_roberta_model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)

        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=num_labels)

    def move_to_device(self, device):
        self.bert_model = self.bert_model.to(device)
        self.xlm_roberta_model = self.xlm_roberta_model.to(device)
        self.distilbert_model = self.distilbert_model.to(device)
        return self

    def forward(self, bert_input_ids, xlmr_input_ids, distil_input_ids):
        bert_logits = self.bert_model(bert_input_ids).logits
        xlm_roberta_logits = self.xlm_roberta_model(xlmr_input_ids).logits
        distilbert_logits = self.distilbert_model(distil_input_ids).logits

        all_logits = torch.stack((bert_logits, xlm_roberta_logits, distilbert_logits), dim=0)
        return all_logits

def load_model(model_path):
    model = torch.load(model_path)
    return model

def predict(text_list, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    max_length = 256

    # Load the custom model and move it to the appropriate device
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()  # Ensure the model is in evaluation mode

    # Tokenize the input texts for each model
    bert_input = model.bert_tokenizer(text_list, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").to(device)
    xlmr_input = model.xlm_roberta_tokenizer(text_list, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").to(device)
    distil_input = model.distilbert_tokenizer(text_list, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt").to(device)
    
    # Run inference and collect predictions
    with torch.no_grad():
        all_logits = model(bert_input["input_ids"], xlmr_input["input_ids"], distil_input["input_ids"])
        predictions = torch.argmax(all_logits, dim=-1)  # Get the index with the highest logit value as the prediction
    
    # Return predictions after moving them to CPU
    return predictions.cpu().numpy().tolist()


