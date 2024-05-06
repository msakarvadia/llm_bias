import sys
sys.path.append('../../')
import argparse
import json
from src.models.model_factory import get_model
from src.utils.initialization import SafeOpen
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

from transformers import (set_seed,
                            TrainingArguments,
                            Trainer,
                            GPT2Config,
                            AutoTokenizer,
                            GPT2Tokenizer,
                            AdamW, 
                            get_linear_schedule_with_warmup,
                            GPT2ForSequenceClassification)


def read_label(inpath, label_type="income"):
    income_dict = {
            "no": 0,
            "low": 1,
            "medium": 2,
            "middle": 2,
            "high" : 3,
            "very high" : 4,
            }
    relationship_dict = {
            "no relation": 0,
            "in relation": 1,
            "married": 2,
            "divorced" : 3,
            "single" : 4,
            }
    sex_dict = {
            "male": 0,
            "female": 1,
            "no valid": 2,
            }
    education_dict = {
            "no highschool": 0,
            "in highschool": 1,
            "hs diploma": 2,
            "in college" : 3,
            "college degree" : 4,
            "phd" : 5,
            }
    indices = []
    labels = []
    with SafeOpen(inpath) as infile:
        count = 0
        for line in tqdm(infile.lines, desc="Reading label",position=0):
            d = json.loads(line)
            pii = d["reviews"]["synth"]
            assert len(pii) == 1, "expected only one pii"
            pii_type = list(pii.keys())[0]
            label = (d["evaluations"]["guess_label"])
            if pii_type == "income" and label_type=="income":
                #print("INCOME")
                #print(label)
                #print(income_dict.keys())
                assert label in income_dict.keys()
                indices.append(count)
                labels.append(income_dict[label])
            if pii_type == "married" and label_type=="marries":
                #print("MARRIED")
                #print(label)
                #print(relationship_dict.keys())
                assert label in relationship_dict.keys()
                indices.append(count)
                labels.append(relationship_dict[label])
            if pii_type == "gender" and label_type=="gender":
                #print("GENDER")
                assert label in sex_dict.keys()
                indices.append(count)
                labels.append(sex_dict[label])
            if pii_type == "education" and label_type=="education":
                #print("EDUCATOIN")
                assert label in education_dict.keys()
                indices.append(count)
                labels.append(education_dict[label])
            count += 1
        num_labels = len(set(labels))
        return labels, indices, num_labels


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

    #load human labels
    print("human labeled eval path: ", cfg.gen_human_labels)
    label_type = "income"
    labels, indices, num_labels = read_label(cfg.gen_human_labels, label_type)

    #Get gpt2 for sequence classification
    model_name_or_path = "gpt2"

    # Get model configuration.
    print('Loading configuraiton...')
    print("num labels: ", num_labels)
    model_config = GPT2Config(num_labels=num_labels,
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

    
    #load embeddings
    generation_embeds = torch.load(cfg.gen_embeds)
    print(len(indices))
    print(len(generation_embeds))
    generation_embeds_current = [ generation_embeds[i] for i in indices]

    assess_device_memory()

    for embed, label in zip(generation_embeds_current, labels):
        print("label: ", label)

        #labels = torch.Tensor([[1, 0, 1]]).to(device) #one hot labels?
        idx = int(label)
        labels = torch.nn.functional.one_hot(torch.tensor(idx), num_classes = num_labels)
        labels = torch.unsqueeze(labels, 0).to(torch.float).to(device)
        print("label: ", labels)
        print("input shape: ", embed.shape)
        #labels = torch.Tensor([[1, 0, 0, 0, 0]]).to(device) #one hot labels?
        #print("working label: ", labels)
        output = seq_model(inputs_embeds=embed, labels=labels) #labels are one-hot
        print("classifier output label: ", torch.argmax(output.logits[0]))
