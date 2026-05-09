import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, XLMRobertaModel, DistilBertModel
from transformers import BertForSequenceClassification, XLMRobertaForSequenceClassification, DistilBertForSequenceClassification, XLMRobertaTokenizer, DistilBertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

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
        

def load_model():
    model = CustomBERTModel(num_labels=2).to(device)
    
    model.load_state_dict(torch.load('class_best_model.pth'))
    
    model.eval()
    return model

def predict(text_list):
    # Load hyperparameters (You can also use get_hyperparams() if needed)
    hyperparams = {'learning_rate': 6.871835594398317e-05, 'batch_size': 66, 'weight_decay': 0.000152125727160186}
    
    model = load_model()

    # Tokenize for each model
    bert_input = model.bert_tokenizer(text_list, truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].to(device)
    xlmr_input = model.xlm_roberta_tokenizer(text_list, truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].to(device)
    distil_input = model.distilbert_tokenizer(text_list, truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        bert_outputs = model.bert_model(bert_input).logits
        xlmr_outputs = model.xlm_roberta_model(xlmr_input).logits
        distil_outputs = model.distilbert_model(distil_input).logits

    # For demonstration, I'm averaging logits. Adjust based on your ensemble strategy.
    average_logits = (bert_outputs + xlmr_outputs + distil_outputs) / 3
    predictions = torch.argmax(average_logits, dim=1).cpu().numpy().tolist()
    
    return predictions



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



# Hyperparameters and Training Setup
best_params = {'learning_rate': 6.871835594398317e-05, 'batch_size': 66, 'weight_decay': 0.000152125727160186}
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]
num_epochs = 10
model = CustomBERTModel(num_labels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=best_params["weight_decay"])
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(X_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
scaler = amp.GradScaler()
train_losses = []
test_losses = []  # Storing test losses
# ... (remaining training and evaluation code)
