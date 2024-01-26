import sys
sys.path.append('../')
from utils import load_quantized_model_and_tokenizer, assess_device_memory
from data.preprocess_data import load_data_from_csv
import os.path
import pandas as pd

from transformers import LlamaConfig
import torch
from torch import nn
from typing import Optional, Tuple
import warnings

from transformers.models.llama.modeling_llama import (LlamaFlashAttention2, 
                                                      LlamaSdpaAttention, 
                                                      LlamaAttention,
                                                      LlamaMLP,
                                                      LlamaRMSNorm )

models = [
"meta-llama/Llama-2-7b-hf",
#"meta-llama/Llama-2-70b-chat-hf",
#"meta-llama/Llama-2-70b-hf",
#"meta-llama/Llama-2-7b-chat-hf",
#"meta-llama/Llama-2-13b-chat-hf",
#"meta-llama/Llama-2-13b-hf",
]

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}

"""LLamaDecoderLayer"""
class LlamaDecoderLayer_Modified(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #self.attn_activations = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        #self.attn_activations = hidden_states
        #print("Dim of attn activations: ", self.attn_activations.shape)

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

if __name__=="__main__":

    #Load data
    data = load_data_from_csv("../data/inference.csv")
    results_file_path = "initial_inference_results.csv"
    if os.path.isfile(results_file_path):
        results = load_data_from_csv(results_file_path)
    else:
        results = data.copy()

    #iterate over models
    for m in models:
        model_name = m

        #check if we have already started processing for a given model
        if not model_name in results.columns:
            results[model_name] = "" #make empty column for results from this model

        #Load model, tokenizer
        model, tokenizer = load_quantized_model_and_tokenizer(model_name)
        assess_device_memory()

        print(model.config)

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        print("did inference part 1")

        # Assign new decoder layer:
        layer_idx = 3
        print(type(model.model.layers[layer_idx]))
        model.model.layers[layer_idx] = LlamaDecoderLayer_Modified(model.config, layer_idx)

        print("Did assing new layer")

        outputs = model(**inputs)
       
        #print(outputs['hidden_states'][layer_idx].shape) # This is outputing residual stream at given layer
        #print("model functions:", dir(model.model.layers[layer_idx].self_attn))
        #print("model variables:", vars(model.model.layers[layer_idx].self_attn.forward)) #This is the function that returns a layers Attention hidden states
        #print(outputs['attentions'][layer_idx].shape) #This is the key, value attention heat map


        '''
        #add column to csv for model name, append generated text to that column
        for index, row in data.iterrows():
            #skip rows that already have content
            if not pd.isna(results.loc[index, model_name]):
                continue
            prompt = row['prompt']
            #better understand what layer_idx = 3 generate text produces TODO
            text = generate_text(text=prompt, model=model, tokenizer=tokenizer)
            results.loc[index, model_name] = text
            if index%50 == 0:
                print(results.tail())
                #Save results
                save_data_as_csv(df=results, filename=results_file_path)
        '''