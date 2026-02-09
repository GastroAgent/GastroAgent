
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch_mul import LlavaMetaModel, LlavaMetaForCausalLM, LlavaMetaForCausalLM

class LlavaConfig(Qwen2Config):
    model_type = "llava_qwen2"
    delay_load = False
    pad_token_id = 151643 # -100
    loss_denorm = None
    
class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / (num_items_in_batch)
    return loss

def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=kwargs.pop('pad_token_id', ignore_index))
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    if kwargs.pop('attention_mask', None) is not None:
        raise NotImplementedError
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss

def find_subtensor_index_vectorized(main_tensor, sub_tensor):
    len_main = main_tensor.size(0)
    len_sub = sub_tensor.size(0)
    windows = main_tensor.unfold(0, len_sub, 1)
    matches = (windows == sub_tensor).all(dim=1)
    if matches.any():
        return matches.nonzero().squeeze().item()
    else:
        return -1

class PloyLlavaLlamaForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        #self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        self.loss_function = ForCausalLMLoss # 使用 自定义的Loss Function。
        # Initialize weights and apply final processing
        self.post_init()


    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            images = kwargs.pop("pixel_values", None) if images is None else images
            batch_num_images = kwargs.get('batch_num_images', None) or kwargs.get('num_images', None)
            
            if isinstance(batch_num_images, int):
                batch_num_images = [batch_num_images]
            # try:
            #     split = [151645, 198, 151644, 77091, 198]
            #     system_user_lengths = [find_subtensor_index_vectorized(x, torch.tensor(split, dtype=x.dtype, device=x.device)) for x in input_ids]
            # except:
            #     system_user_lengths = -1
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                batch_num_images = batch_num_images,
                # system_user_lengths=system_user_lengths
            )

        # print('inputs_embeds requires_grad: ', inputs_embeds.requires_grad)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            logits_to_keep = logits_to_keep,
            return_dict=return_dict,
            pad_token_id=getattr(self.config, 'pad_token_id', -100),
            ### # when use pad in label and image_placeholder is -100 in label. (Unmasked Targets)
            # num_items_in_batch = ((labels != -100) & attention_mask).sum() if labels is not None else None
            ### # when use ignore_index in label. (Masked Targets using Ignore_index)
            num_items_in_batch = getattr(self.config, 'loss_denorm', kwargs.pop('num_items_in_batch', None)) # Full-unbias 的 loss 计算。
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        inputs = kwargs.pop("input_ids", None) if inputs is None else inputs
        images = kwargs.pop("pixel_values", None) if images is None else images
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            batch_num_images = kwargs.pop('batch_num_images', None) or kwargs.pop('num_images', None)
            if isinstance(batch_num_images, int):
                batch_num_images = [batch_num_images]
            
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                batch_num_images=batch_num_images,
                
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        inputs.pop("cache_position")
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_qwen2", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, PloyLlavaLlamaForCausalLM)
# AutoModel.register(LlavaConfig, PloyLlavaLlamaForCausalLM)