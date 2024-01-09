import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def assess_device_memory():
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    print("free GB:", free_in_GB)

def load_model_w_bits_and_bytes(model_name):
    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map='auto',
      load_in_4bit=True,
      max_memory=max_memory
    )
    return model

def load_model_w_HF_device_map(model_name):
    #TODO
    
    return model

MAX_NEW_TOKENS = 128
model_name = "mistralai/Mistral-7B-v0.1"
model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-13b-chat-hf"
model_name = "meta-llama/Llama-2-70b-chat-hf"

text = 'Hamburg is in which country?\n'
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

assess_device_memory()
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
print("max mem:", max_memory)

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}


load_model_w_bits_and_bytes(model_name):

print("Loaded model into memory")
assess_device_memory()

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
print("This is the model's input text: ", text)
print("This is the model's response: ")
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))




