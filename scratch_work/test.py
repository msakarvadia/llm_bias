from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch

def assess_device_memory():
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    print("free GB:", free_in_GB)

assess_device_memory()

name = "meta-llama/Llama-2-7b-chat-hf"
name = "meta-llama/Llama-2-13b-chat-hf"
name = "meta-llama/Llama-2-70b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

assess_device_memory()

text = "Tell me a funny story:  "    # prompt goes here

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
