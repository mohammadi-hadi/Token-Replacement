import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, XLMRobertaModel, DistilBertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hyperparams():
    with open("best_params (1).json", "r") as f:
        hyperparams = json.load(f)
    return hyperparams

class CustomBERTModel(nn.Module):
    def __init__(self, num_labels, dropout_rate):
        super(CustomBERTModel, self).__init__()
        
        # Loading models
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.xlm_roberta_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        
        # Dropout layer with dynamic dropout rate
        self.dropout = nn.Dropout(dropout_rate)
        
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

def load_model(dropout_rate):
    model = CustomBERTModel(num_labels=2, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load("best_model (1).pth"))
    model.eval()
    
    return model

def predict(text_list):
    # Load hyperparameters
    hyperparams = get_hyperparams()
    dropout_rate = hyperparams.get("dropout_rate", 0.1)  # Use 0.1 as default if not present in json
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    encodings = tokenizer(text_list, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    
    model = load_model(dropout_rate)
    
    with torch.no_grad():
        outputs = model(input_ids)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
    
    return predictions
