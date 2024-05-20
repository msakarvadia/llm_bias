import sys

sys.path.append("../../")
import argparse
import gc
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
from utils import (
    load_quantized_model_and_tokenizer,
    assess_device_memory,
    load_distributed_model_and_tokenizer,
)
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
from src.prompts.make_prompts import create_prompts

from transformers import (
    set_seed,
    TrainingArguments,
    Trainer,
    GPT2Config,
    AutoTokenizer,
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    GPT2ForSequenceClassification,
)


def read_label(inpath, label_type="income"):
    income_dict = {
        "no": 0,
        "low": 1,
        "medium": 2,
        "middle": 2,
        "high": 3,
        "very high": 4,
    }
    relationship_dict = {
        "no relation": 0,
        "in relation": 1,
        "married": 2,
        "divorced": 3,
        "single": 4,
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
        "in college": 3,
        "college degree": 4,
        "phd": 5,
    }
    indices = []
    labels = []
    with SafeOpen(inpath) as infile:
        count = 0
        for line in tqdm(infile.lines, desc="Reading label", position=0):
            d = json.loads(line)
            pii = d["reviews"]["synth"]
            assert len(pii) == 1, "expected only one pii"
            pii_type = list(pii.keys())[0]
            label = d["evaluations"]["guess_label"]
            if pii_type == "income" and label_type == "income":
                assert label in income_dict.keys()
                indices.append(count)
                labels.append(income_dict[label])
                num_labels = len(set(income_dict.values()))
            if pii_type == "married" and label_type == "married":
                assert label in relationship_dict.keys()
                indices.append(count)
                labels.append(relationship_dict[label])
                num_labels = len(set(relationship_dict.values()))
            if pii_type == "gender" and label_type == "gender":
                assert label in sex_dict.keys()
                indices.append(count)
                labels.append(sex_dict[label])
                num_labels = len(set(sex_dict.values()))
            if pii_type == "education" and label_type == "education":
                assert label in education_dict.keys()
                indices.append(count)
                labels.append(education_dict[label])
                num_labels = len(set(education_dict.values()))
            count += 1
        # num_labels = len(set(labels))
        return labels, indices, num_labels


def check_gc():
    num_objects = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                # print(type(obj), obj.size())
                num_objects += 1
        except:
            pass
    print("NUM OBJECTS IN GPU MEMORY BY GC: ", num_objects)


def extract_embeds():
    return 0


def train(
    model,
    seq_model,
    prompts,
    labels_original,
    optimizer,
    epochs,
    mask,
    desired_label,
    num_labels,
):
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 1
    labels = torch.FloatTensor(batch_size, num_labels).to(device)
    labels.zero_()
    # labels = torch.nn.functional.one_hot(
    #    torch.tensor(0), num_classes=num_labels
    # )
    # labels = torch.unsqueeze(1, labels, 0).to(torch.float).to(device)
    # print("label shape: ", labels.shape)

    for i in range(epochs):
        predictions = []
        avg_loss = 0
        for batch, label in tqdm(zip(prompts, labels_original)):
            labels.zero_()
            optimizer.zero_grad()
            model.model.zero_grad(set_to_none=True)
            seq_model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary())
            assess_device_memory()
            check_gc()
            # with torch.enable_grad():
            # print(batch.get_prompt())
            results, hidden_states, input_len = model.predict_logits_w_mask(batch, mask)
            # new_generation_hidden_states = hidden_states[0][-1][:, -1, :]
            # print(
            #    "new inputs shape: ",
            #    new_generation_hidden_states.unsqueeze_(dim=1).shape,
            # )
            hs = []
            for token in range(len(hidden_states)):
                layer_num = -1
                hs.append(hidden_states[token][layer_num][:, -1, :])
            print(hidden_states[-1][-1][:, -1, :])
            # print(hidden_states[-1][-1][:,-1,:])

            # TODO (make a placeholder tensor rather than new one every single time
            inputs = torch.stack(hs, dim=1)  # .to(device)
            print("working inputs shape: ", inputs.shape)

            labels[0, desired_label] = 1
            print("original label: ", label)
            print("desired labels: ", labels)
            output = seq_model(
                inputs_embeds=inputs, labels=labels
            )  # labels are one-hot
            # inputs.detach_()
            # labels.detach_()
            del inputs
            del hidden_states
            # del labels
            predictions.append(torch.argmax(output.logits[0]).item())
            print("classifier output prediction: ", torch.argmax(output.logits[0]))
            print("desired label: ", desired_label)

            loss = criterion(output.logits, desired_label) - output.loss
            print("classifier output loss: ", loss)
            loss.backward()
            # output.loss.backward()
            # print("Gradients: ", mask.grad)
            # print("Mask: ", mask)
            optimizer.step()
            # avg_loss += output.loss
            print("_______________________________")

        desired_labels = [desired_label.item()] * len(labels_original)
        print("desired_labels: ", desired_labels)
        print("predictions: ", predictions)
        print("num data points: ", len(labels_original))
        print("Avg Loss: ", avg_loss / len(labels_original))
        train_acc = accuracy_score(desired_labels, predictions)
        print("Train acc: ", train_acc)

    return 0  # predictions


def eval_model(model, seq_model, inputs, labels_original, desired_label):
    print("Evaluating model-------")
    seq_model.eval()
    batch_size = 1
    labels = torch.FloatTensor(batch_size, num_labels).to(device)
    labels.zero_()
    labels[0, desired_label] = 1
    with torch.no_grad():
        predictions = []
        avg_loss = 0
        for batch, label in tqdm(zip(prompts, labels_original)):
            results, hidden_states, input_len = model.predict_logits_w_mask(batch, mask)
            hs = []
            for token in range(len(hidden_states)):
                layer_num = -1
                hs.append(hidden_states[token][layer_num][:, -1, :])

            inputs = torch.stack(hs, dim=1)  # .to(device)

            output = seq_model(
                inputs_embeds=inputs, labels=labels
            )  # labels are one-hot
            predictions.append(torch.argmax(output.logits[0]).item())

        print("num data points: ", len(labels_original))
        print("Eval Avg Loss: ", avg_loss / len(labels_original))
        train_acc = accuracy_score(labels_original, predictions)
        print("Eval acc: ", train_acc)
        return
    return 0


if __name__ == "__main__":
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
    # f, path = get_out_file(cfg)

    # Load data
    print("loading data...")
    synthetic_data_path = "../../data/synthetic_dataset.jsonl"
    profiles = load_data(synthetic_data_path)
    print(profiles[0].comments)

    # Create prompts
    prompts = []
    for profile in profiles:
        prompt = create_prompts(profile)
        prompts += prompt

    # load human labels
    print("human labeled eval path: ", cfg.gen_human_labels)
    label_type = "income"
    labels, indices, num_labels = read_label(cfg.gen_human_labels, cfg.demographic)

    # Load model to defined device.
    seq_model = torch.load(cfg.discrim_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_model.to(device)
    seq_model.eval()
    print("Model loaded to `%s`" % device)

    # load embeddings
    generation_embeds = torch.load(cfg.gen_embeds)
    generation_embeds_current = [generation_embeds[i] for i in indices]
    prompts = [prompts[i] for i in indices]

    # TODO (MS) temporarily shortening the dataset to three samples:
    # prompts = prompts[7:11]
    # labels = labels[7:11]
    # generation_embeds_current = generation_embeds_current[:, :3]

    assess_device_memory()

    logging.getLogger("transformers").setLevel(logging.ERROR)
    print(labels)
    model = get_model(cfg.gen_model)
    mask = torch.rand(
        (1, 5, 4096),
        requires_grad=True,
        device=device,
        dtype=torch.bfloat16,  # mask for 5 token positions
    )
    # mask.retain_grad()
    print("mask before optim: ", mask)
    optimizer = torch.optim.AdamW(
        [mask],
        lr=0.5,  # default is 5e-5, our notebook had 2e-5 #lr = 2 is too high
        eps=1e-8,  # default is 1e-8.
    )

    x_train, x_test, y_train, y_test = train_test_split(
        prompts, labels, test_size=0.1, random_state=cfg.seed
    )
    desired_label = torch.tensor([1], device=device)

    eval_model(model, seq_model, x_test, y_test, desired_label)

    predictions = train(
        model,
        seq_model,
        x_train,
        y_train,
        optimizer,
        cfg.epochs,
        mask,
        desired_label,
        num_labels,
    )
    torch.save(mask, "gender_mask_female.pt")

    eval_model(model, seq_model, x_test, y_test, desired_label)
