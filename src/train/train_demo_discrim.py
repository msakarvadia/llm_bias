import sys
sys.path.append('../../')
import argparse
import json
from sklearn.model_selection import train_test_split
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
import logging

from transformers import LlamaConfig
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import warnings
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

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
                assert label in income_dict.keys()
                indices.append(count)
                labels.append(income_dict[label])
                num_labels = len(set(income_dict.values()))
            if pii_type == "married" and label_type=="married":
                assert label in relationship_dict.keys()
                indices.append(count)
                labels.append(relationship_dict[label])
                num_labels = len(set(relationship_dict.values()))
            if pii_type == "gender" and label_type=="gender":
                assert label in sex_dict.keys()
                indices.append(count)
                labels.append(sex_dict[label])
                num_labels = len(set(sex_dict.values()))
            if pii_type == "education" and label_type=="education":
                assert label in education_dict.keys()
                indices.append(count)
                labels.append(education_dict[label])
                num_labels = len(set(education_dict.values()))
            count += 1
        #num_labels = len(set(labels))
        return labels, indices, num_labels

def train(model, inputs, labels_original, optimizer, epochs):
    model.train()

    for i in range(epochs):
        predictions = []
        avg_loss = 0
        for batch, label in tqdm(zip(inputs, labels_original)):
            model.zero_grad()

            idx = int(label)
            labels = torch.nn.functional.one_hot(torch.tensor(idx), num_classes = num_labels)
            labels = torch.unsqueeze(labels, 0).to(torch.float).to(device)
            output = seq_model(inputs_embeds=batch, labels=labels) #labels are one-hot
            predictions.append(torch.argmax(output.logits[0]).item())
            #print("classifier output label: ", torch.argmax(output.logits[0]))
            #print("classifier output loss: ", output.loss)

            output.loss.backward()
            avg_loss += output.loss
            optimizer.step()

        print("num data points: ", len(labels_original))
        print("Avg Loss: ", avg_loss / len(labels_original))
        train_acc = accuracy_score(labels_original, predictions)
        print("Train acc: ", train_acc)
        
    return predictions

def eval_model(model, inputs, labels_original):
    print("Evaluating model-------")
    model.eval()
    with torch.no_grad():
        predictions = []
        avg_loss = 0
        for batch, label in tqdm(zip(inputs, labels_original)):

            idx = int(label)
            labels = torch.nn.functional.one_hot(torch.tensor(idx), num_classes = num_labels)
            labels = torch.unsqueeze(labels, 0).to(torch.float).to(device)
            output = seq_model(inputs_embeds=batch, labels=labels) #labels are one-hot
            predictions.append(torch.argmax(output.logits[0]).item())

            avg_loss += output.loss

        print("num data points: ", len(labels_original))
        print("Eval Avg Loss: ", avg_loss / len(labels_original))
        train_acc = accuracy_score(labels_original, predictions)
        print("Eval acc: ", train_acc)
        return
    return 0

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
    labels, indices, num_labels = read_label(cfg.gen_human_labels, cfg.demographic)

    #Get gpt2 for sequence classification
    model_name_or_path = "gpt2"

    # Get model configuration.
    print('Loading configuration...')
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
    generation_embeds_current = [ generation_embeds[i] for i in indices]

    assess_device_memory()

    optimizer = torch.optim.AdamW(seq_model.parameters(),
                                        lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                                        eps = 1e-8 # default is 1e-8.
                                        )

    logging.getLogger("transformers").setLevel(logging.ERROR)
    print(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(generation_embeds_current, labels, test_size=0.1, random_state=cfg.seed)
    eval_model(seq_model, x_test, y_test)
    predictions = train(seq_model, x_train, y_train, optimizer, cfg.epochs)
    eval_model(seq_model, x_test, y_test)

    torch.save(seq_model, f'{cfg.demographic}_discrim_model.pt')

    #predictions = train(seq_model, generation_embeds_current, labels, optimizer, cfg.epochs)
