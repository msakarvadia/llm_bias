from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline


# Download the Weights
checkpoint = "meta-llama/Llama-2-70b-chat-hf"
#weights_location = hf_hub_download(checkpoint)
weights_location = "/net/scratch/sakarvadia/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2"

# Create a model and initialize it with empty weights
config = AutoConfig.from_pretrained(checkpoint)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Load the checkpoint and dispatch it to the right devices
model = load_checkpoint_and_dispatch(
    model, weights_location, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"]
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)


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