import os
import json
import torch
from transformers import InstructBlipForConditionalGeneration
from transformers import Blip2ForConditionalGeneration
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from transformers import LlavaForConditionalGeneration

from peft import LoraConfig
from peft import LoftQConfig
from peft import get_peft_model
from peft import PeftModel

from .utils import convert_models_to_fp32

__all__ = ['get_model']


def get_model(cnf):
    if cnf.architecture == 'BLIP':
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.bfloat16
        )
        for param in model.parameters():
            param.requires_grad = True
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    elif cnf.architecture == 'InBLIPxl':
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            torch_dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    elif cnf.architecture == 'InBLIPxxl':
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xxl",
            torch_dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
    elif cnf.architecture == "LlaVa":
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    elif cnf.architecture == "LlaVa+":
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        peft_model_id = cnf.peft_weights
        model.vision_tower = PeftModel.from_pretrained(model.vision_tower, peft_model_id)
    else:
        raise NotImplementedError
    if not cnf.finetune:
        for name, param in model.named_parameters():
            if 'head' in name or "query_tokens" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True

    #if processor.tokenizer.pad_token_id is None:  # set pad token
    #        processor.tokenizer.pad_token_id = 0
    return model, processor
