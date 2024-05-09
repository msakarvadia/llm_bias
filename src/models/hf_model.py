from typing import List, Dict, Iterator, Tuple
import torch
from src.configs import ModelConfig
from src.prompts import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import BaseModel

from undecorated import undecorated
from types import MethodType


class HFModel(BaseModel):
    curr_models: Dict[str, AutoModelForCausalLM] = {}

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if (
            config.name not in self.curr_models
        ):  # Hacky but avoids loading the same model multiple times
            if config.dtype == "float16":
                dtype = torch.float16
            if config.dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            self.curr_models[config.name] = AutoModelForCausalLM.from_pretrained(
                config.name,
                load_in_8bit=True,  # NOTE (MS): add this in for larger model
                trust_remote_code=True,
                torch_dtype=dtype,
                # torch_dtype=(
                #    torch.float16 if config.dtype == "float16" else torch.float32
                # ),
                device_map=config.device,
            )
        self.model: AutoModelForCausalLM = self.curr_models[config.name]
        # self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        #     config.name,
        #     torch_dtype=torch.float16 if config.dtype == "float16" else torch.float32,
        #     device_map=config.device,
        # )
        self.device = self.model.device
        if config.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def predict_logits_w_mask(self, input: Prompt, mask, **kwargs):
        text = input.get_prompt().rstrip()

        model_text = self.apply_model_template(text)

        input_ids = self.tokenizer.encode(
            model_text,
            return_tensors="pt",
        ).to(self.device)
        input_length = len(input_ids[0])
        # print("input prompt length: ", input_length)

        # NOTE (MS): the generate function is decorated w/ torch.no_grad so to work around it do https://discuss.huggingface.co/t/how-to-output-loss-from-model-generate/16999/12
        with torch.no_grad():
            output_1 = self.model.generate.__wrapped__(
                self.model,
                input_ids,
                return_dict_in_generate=True,
                output_logits=False,
                output_hidden_states=True,
                max_new_tokens=1,
            )
        embeds = output_1.hidden_states[0][-1]
        print("mask shape: ", mask.shape)
        print("Shape of original prompt embeddings: ", embeds.shape)

        embeds = torch.cat((mask, embeds), dim=1)
        output = self.model.generate.__wrapped__(
            self.model,
            inputs_embeds=embeds,
            return_dict_in_generate=True,
            output_logits=False,
            output_hidden_states=True,
            **self.config.args
        )

        # For decoder only models:
        out_ids = output.sequences[:, input_length:]

        return (
            self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip(),
            output.hidden_states,
            input_length,
        )

    def predict_logits(self, input: Prompt, **kwargs):
        text = input.get_prompt().rstrip()

        model_text = self.apply_model_template(text)

        input_ids = self.tokenizer.encode(
            model_text,
            return_tensors="pt",
        ).to(self.device)
        input_length = len(input_ids[0])

        output = self.model.generate(
            input_ids,
            return_dict_in_generate=True,
            output_logits=False,
            output_hidden_states=True,
            **self.config.args
        )
        # print("TOKEN INPUT LENGTH: ", input_length)
        # print("produced geneartions shape: ", output.sequences.shape)
        # print("num of hidden states: ", len(output.hidden_states)) #hidden states (# of new tokens, # of layers, )
        # print("# of layers in hidden states len: ", len(output.hidden_states[-1])) #hidden states (# of new tokens, # of layers, )
        # print("hidden states shape of last new token at last layer: ", output.hidden_states[-1][-1].shape)
        # for i in range(5):
        #    print("token index: ", i)
        #    print("hidden states shape of last i'th token at last layer: ", output.hidden_states[i][-1].shape)

        # For decoder only models:
        out_ids = output.sequences[:, input_length:]

        return (
            self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip(),
            output.hidden_states,
            input_length,
        )

    def predict(self, input: Prompt, **kwargs):
        text = input.get_prompt().rstrip()

        model_text = self.apply_model_template(text)

        input_ids = self.tokenizer.encode(
            model_text,
            return_tensors="pt",
        ).to(self.device)
        input_length = len(input_ids[0])

        output = self.model.generate(input_ids, **self.config.args)

        # For decoder only models:
        out_ids = output[:, input_length:]

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        new_inputs = []
        for input in inputs:
            text = input.get_prompt().rstrip()
            new_inputs.append(self.apply_model_template(text))

        if "max_workers" in kwargs:
            batch_size = kwargs["max_workers"]
        else:
            batch_size = 1

        for i in range(0, len(new_inputs), batch_size):
            end = min(i + batch_size, len(new_inputs))
            new_inputs_batch = new_inputs[i:end]

            model_inputs = self.tokenizer(
                new_inputs_batch,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(self.device)
            input_length = len(model_inputs["input_ids"][0])

            output = self.model.generate(**model_inputs, **self.config.args)

            outs_str = self.tokenizer.batch_decode(
                output[:, input_length:], skip_special_tokens=True
            )
            for j in range(len(outs_str)):
                yield (inputs[i + j], outs_str[j])
            # yield (inputs[i], outs_str[0])

            # results.extend(outs_str)

        # return results

    def predict_string(self, input: str, **kwargs) -> str:
        return self.predict(Prompt(intermediate=input), **kwargs)
