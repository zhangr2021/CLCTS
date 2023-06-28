import argparse
import logging
import os
import copy
import sentencepiece

from longformer_encoder_decoder import LongformerSelfAttentionForMBart 
from longformer_encoder_decoder import LongformerEncoderDecoderConfig
from longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch import Tensor

from transformers import MBart50Tokenizer, MBartTokenizer
from transformers import MBartForConditionalGeneration, AutoTokenizer

from typing import List, Optional, Tuple, Dict
from torch import nn, Tensor
from transformers import LongformerSelfAttention
from transformers import MBartConfig, MBartForConditionalGeneration

from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
)
import pandas as pd
import numpy as np
import nltk


def create_long_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    attention_window,
    max_pos
):

    model = MBartForConditionalGeneration.from_pretrained(base_model, cache_dir=cache_dir)
    tokenizer = MBart50Tokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos, cache_dir=cache_dir) #, src_lang="de_DE", tgt_lang="en_XX"
    config = LongformerEncoderDecoderConfig.from_pretrained(base_model)

    model.config = config

    config.max_seq_len = max_seq_len
    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = ['LongformerEncoderDecoderForConditionalGeneration', ]

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    assert current_max_pos == config.max_position_embeddings + 2

    config.max_encoder_position_embeddings = max_pos
    config.max_decoder_position_embeddings = config.max_position_embeddings
    del config.max_position_embeddings
    max_pos += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    assert max_pos >= current_max_pos

    # allocate a larger position embedding matrix for the encoder
    new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

    # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_bart = LongformerSelfAttentionForMBart(config, layer_id=i)

        longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_bart.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
        longformer_self_attn_for_bart.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
        longformer_self_attn_for_bart.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

        longformer_self_attn_for_bart.output = layer.self_attn.out_proj

        layer.self_attn = longformer_self_attn_for_bart

    # save model
    print(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

    
#save_model_to
base_model = 'facebook/mbart-large-50-many-to-many-mmt'
tokenizer_name_or_path = 'facebook/mbart-large-50-many-to-many-mmt'

save_model_to = "/mLED_model"
cache_dir = "/cache"



max_seq_len = 4096

model, tokenizer = create_long_model(
        save_model_to=save_model_to,
        base_model=base_model,
        tokenizer_name_or_path=tokenizer_name_or_path,
        attention_window= 512,
        max_pos=4096
    )



