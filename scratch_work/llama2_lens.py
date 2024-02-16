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
"EleutherAI/gpt-j-6B",
#"meta-llama/Llama-2-7b-hf",
#"meta-llama/Llama-2-13b-chat-hf",
#"meta-llama/Llama-2-13b-hf",
"gpt-neo-125M",
"EleutherAI/gpt-neo-1.3B",
"EleutherAI/gpt-neo-2.7B",
]

import torch

if __name__=="__main__":


    #iterate over models
    for m in models:
        model_name = m


        #Load model, tokenizer
        assess_device_memory()

        '''
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        hf_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16
                                                   )

        #Load model into TransformerLens template
        model = HookedTransformer.from_pretrained(model_name, hf_model=hf_model, tokenizer=tokenizer, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)
        '''

        model = HookedTransformer.from_pretrained(model_name, device="cuda")
        print(model)
        #model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        #print("model moved to GPU")
        assess_device_memory()
        prompt = "disabled. BCG.com will work better for you if you enable JavaScript or switch to a JavaScript supported browser. Boston Consulting Group is an Equal Opportunity Employer. All qualified applicants will receive consideration for employment without regard to race, color,"
        prompt = "use this file except in compliance with the License. * You may obtain a copy of the License at * http://www.apache.org/licenses/LICENSE-2.0 * Unless required by applicable law or agreed"
        output = model.generate(prompt, max_new_tokens=50, temperature=0)
        print(output)
