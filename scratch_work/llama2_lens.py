import sys
sys.path.append('../')
from utils import load_quantized_model_and_tokenizer, assess_device_memory, generate_text
from data.preprocess_data import load_data_from_csv, save_data_as_csv
import os.path
import pandas as pd

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer

from transformers import LlamaForCausalLM, LlamaTokenizer

models = [
#"meta-llama/Llama-2-70b-chat-hf",
#"meta-llama/Llama-2-70b-hf",
#"meta-llama/Llama-2-7b-chat-hf",
"meta-llama/Llama-2-7b-hf",
#"meta-llama/Llama-2-13b-chat-hf",
"meta-llama/Llama-2-13b-hf",
]

import torch

if __name__=="__main__":


    #iterate over models
    for m in models:
        model_name = m


        #Load model, tokenizer
        assess_device_memory()

        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        hf_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16
                                                   )

        #Load model into TransformerLens template
        model = HookedTransformer.from_pretrained(model_name, hf_model=hf_model, tokenizer=tokenizer, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)
        print(model)

        #model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        #print("model moved to GPU")
        assess_device_memory()
        output = model.generate("The capital of Germany is", max_new_tokens=20, temperature=0)
        print(output)
