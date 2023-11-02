"""This file contains useful helper functions for reserach."""

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os

def load_and_save_model_from_hub(model_name:str, save_dir:str="models/") -> (AutoModelForCausalLM):
    """This function downloads functions from huggingface and save them locally,
    for llama models you will need a token that proves you have a licence to download """
    print("model name: ", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.save_pretrained(save_dir+model_name, from_pt=True)

    return model



models = [
"meta-llama/Llama-2-7b-chat-hf",
"meta-llama/Llama-2-7b-hf",
"meta-llama/Llama-2-13b-chat-hf",
"meta-llama/Llama-2-13b-hf",
"meta-llama/Llama-2-70b-chat-hf",
"meta-llama/Llama-2-70b-hf",
]
for model_name in models:
    modelr = load_and_save_model_from_hub(model_name)

