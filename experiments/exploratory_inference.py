import sys
sys.path.append('../')
from utils import load_quantized_model_and_tokenizer, assess_device_memory, generate_text

models = [
"meta-llama/Llama-2-7b-chat-hf",
"meta-llama/Llama-2-7b-hf",
"meta-llama/Llama-2-13b-chat-hf",
"meta-llama/Llama-2-13b-hf",
"meta-llama/Llama-2-70b-chat-hf",
"meta-llama/Llama-2-70b-hf",
]

if __name__=="__main__":
    model, tokenizer = load_quantized_model_and_tokenizer(models[0])
    assess_device_memory()
    generate_text(text="Tell me a sad story.", model=model, tokenizer=tokenizer)