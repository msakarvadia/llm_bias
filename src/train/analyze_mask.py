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


def get_top_k_vocab_from_mask(mask, embed_mat, k, tokenizer):

    transformed_output = mask @ embed_mat.T
    val, top_tokens = torch.topk(transformed_output, dim=2, k=k)
    print(top_tokens)
    print(transformed_output.shape)
    for tokens in top_tokens[0]:
        print(tokenizer.decode(tokens.tolist()))

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

    # Load model to defined device.
    seq_model = torch.load(cfg.discrim_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_model.to(device)
    seq_model.eval()
    print("Model loaded to `%s`" % device)

    # Load LLM
    model = get_model(cfg.gen_model)
    embed_mat = 0
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(name)
        if name == "model.embed_tokens.weight":
            print("embedding shape: ", param.shape)
            embed_mat = param

    # Load trained Mask
    trained_mask = torch.load(cfg.mask_name)

    print(trained_mask)
    print(trained_mask.shape)

    get_top_k_vocab_from_mask(trained_mask, embed_mat, k=100, tokenizer=model.tokenizer)

# Project mask values into vocab space using LLM's embedding matrix:
