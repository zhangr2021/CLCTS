#based on: https://github.com/allenai/longformer/blob/master/longformer/longformer_encoder_decoder.py
#changed with help: https://github.com/Taeksu-Kim/longformer_kobart
#and https://huggingface.co/transformers/v4.2.2/_modules/transformers/models/mbart/modeling_mbart.html

from typing import List, Optional, Tuple, Dict
from torch import nn, Tensor
from transformers import LongformerSelfAttention
from transformers import MBartConfig, MBartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch import Tensor


class LongformerEncoderDecoderForConditionalGeneration(MBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:  
            self.model.encoder.embed_positions = MBartLearnedPositionalEmbedding(config.max_encoder_position_embeddings,  config.d_model)
            self.model.decoder.embed_positions = MBartLearnedPositionalEmbedding(config.max_decoder_position_embeddings, config.d_model)
            
            for i, layer in enumerate(self.model.encoder.layers):
                layer.self_attn = LongformerSelfAttentionForMBart(config, layer_id=i)


class LongformerEncoderDecoderConfig(MBartConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks',
                 gradient_checkpointing: bool = False, **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2']


class LongformerSelfAttentionForMBart(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.longformer_self_attn = LongformerSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
       self,
       hidden_states: torch.Tensor,
       key_value_states: Optional[torch.Tensor] = None,
       past_key_value: Optional[Tuple[torch.Tensor]] = None,
       attention_mask: Optional[torch.Tensor] = None,
       layer_head_mask: Optional[torch.Tensor] = None,
       output_attentions: bool = False,
   ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

       is_cross_attention = key_value_states is not None
       bsz, tgt_len, embed_dim = hidden_states.size()

       attention_mask = attention_mask.squeeze(dim=1)
       attention_mask = attention_mask[:,0]

       is_index_masked = attention_mask < 0
       is_index_global_attn = attention_mask > 0
       is_global_attn = is_index_global_attn.flatten().any().item()

       outputs = self.longformer_self_attn(
           hidden_states,
           attention_mask=attention_mask,
           layer_head_mask=None,
           is_index_masked=is_index_masked,
           is_index_global_attn=is_index_global_attn,
           is_global_attn=is_global_attn,
           output_attentions=output_attentions,
       )

       attn_output = self.output(outputs[0])

       return (attn_output,) + outputs[1:] if len(outputs) == 2 else (attn_output, None, None)