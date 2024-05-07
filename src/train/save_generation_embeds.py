import sys
sys.path.append('../../')
import argparse
from src.models.model_factory import get_model
from src.utils.initialization import (
        read_config_from_yaml,
        seed_everything,
        set_credentials,
        get_out_file,
        )
from utils import load_quantized_model_and_tokenizer, assess_device_memory, load_distributed_model_and_tokenizer
from data.preprocess_data import load_data_from_csv
import os.path
import pandas as pd

from transformers import LlamaConfig
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import warnings
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

from src.utils.reddit_utils import load_data, type_to_str, type_to_options
from src.utils.reddit_types import Profile
from src.prompts.prompt import Prompt
from src.prompts.make_prompts import create_prompts 

from transformers import (set_seed,
                            TrainingArguments,
                            Trainer,
                            GPT2Config,
                            AutoTokenizer,
                            GPT2Tokenizer,
                            AdamW, 
                            get_linear_schedule_with_warmup,
                            GPT2ForSequenceClassification)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/acs_config.yaml",
        help="Path to the config file",
        )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    # Set seed for reproducibility.
    set_seed(cfg.seed)
    set_credentials(cfg)
    #f, path = get_out_file(cfg)

    #Load data
    print("loading data...")
    synthetic_data_path = "../../data/synthetic_dataset.jsonl" 
    profiles = load_data(synthetic_data_path)
    print(profiles[0].comments)

    # Create prompts
    prompts = []
    for profile in profiles:
        prompt = create_prompts(profile)
        prompts += prompt 

    #Get gpt2 for sequence classification
    model_name_or_path = "gpt2"
    n_labels = 3

    # Get model configuration.
    print('Loading configuraiton...')
    model_config = GPT2Config(num_labels=n_labels,
                                      n_embd=4096, # This is side of Llama vocab
                                      n_head=8,
                                      n_layer=4,
                                      )

    # Get the actual model.
    print('Loading model...')
    seq_model = GPT2ForSequenceClassification(model_config)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # resize model embedding to match new tokenizer
    seq_model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    seq_model.config.pad_token_id = seq_model.config.eos_token_id

    # Load model to defined device.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seq_model.to(device)
    print('Model loaded to `%s`'%device)

    
    #Load model, tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_from_other_repo = get_model(cfg.gen_model)

    assess_device_memory()


    last_layer_embedding_list = []
    #model.eval()
    for i in tqdm(prompts):
        #print(i.to_dict())
        prompt = i.get_prompt()
        #print("------------------- MODEL PROMPTS: -----------------")
        #print(prompt)
        #print("------------------- MODEL PROMPT END -----------------")
        with torch.no_grad():
            results, hidden_states, input_len  = model_from_other_repo.predict_logits(i)


        #print("------------------- MODEL GENERATIONS: -----------------")
        #print("New model generation: ", results)
        #print("------------------- MODEL GENERATIONS END -------------- ")
        new_generation_hidden_states = hidden_states[0][-1][:, -1, :]
        hs = []
        for token in range(len(hidden_states)):
            layer_num = -1
            hs.append(hidden_states[token][layer_num][:, -1, :])

        #print("inputing hs into seq model of size: ", torch.stack(hs, dim=1).shape)
        inputs = torch.stack(hs, dim=1) #.to(device)
        last_layer_embedding_list.append(inputs)
        ''' TODO (MS): this works
        labels = torch.Tensor([[1, 0, 1]]).to(device) #one hot labels?
        output = seq_model(inputs_embeds=inputs, labels=labels) #labels are one-hot
        print("classifier output label: ", torch.argmax(output.logits[0]))
        '''

    torch.save(last_layer_embedding_list, "embeddings_llama7b_generation_last_layer_embeddings.pt")
