# =============================
# 1. Experiment Environment Setup
# =============================

import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os

# -----------------------------
# Environment and Random Seeds
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ Running in CPU mode, performance experiments will be slower.")

# -----------------------------
# Hugging Face Model Loading Configuration
# -----------------------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer

# 💡 A Chinese model is more suitable for Chinese input
model_name = "uer/gpt2-chinese-cluecorpussmall"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model {model_name} loaded successfully")