import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/opt-125m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    return model, tokenizer, device