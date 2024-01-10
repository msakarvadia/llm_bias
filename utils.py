"""This file contains useful helper functions for reserach."""

# Load model directly
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch

def load_quantized_model_and_tokenizer(model_name:str) -> (AutoModelForCausalLM, AutoTokenizer):
    """This function downloads functions from huggingface and save them locally,
    for llama models you will need a token that proves you have a licence to download """
    print("model name: ", model_name)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

    return model, tokenizer

def assess_device_memory():
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    print("free GB:", free_in_GB)

def generate_text(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, text:str="Tell me a funny story."):
    generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
    )

    sequences = generation_pipe(
        text,
        max_length=128,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=10,
        temperature=0.4,
        top_p=0.9
    )

    print(sequences[0]["generated_text"])

    return



models = [
"meta-llama/Llama-2-7b-chat-hf",
"meta-llama/Llama-2-7b-hf",
"meta-llama/Llama-2-13b-chat-hf",
"meta-llama/Llama-2-13b-hf",
"meta-llama/Llama-2-70b-chat-hf",
"meta-llama/Llama-2-70b-hf",
]

#for model_name in models:
#    model, tokenizer = load_quantized_model_and_tokenizer(model_name)

