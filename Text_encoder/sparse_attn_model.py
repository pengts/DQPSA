import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import BertModel, BertPreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from fsa import FSA_layer
from epe import EPE

@dataclass
class UIEModelOutput(ModelOutput):
    span_prob:torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    loss_bce: Optional[torch.FloatTensor] = None
    loss_fsl: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    fuzzy_span_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_epe: Optional[torch.FloatTensor] = None
    span_logits:torch.FloatTensor = None


def get_span_for_eval(start_list,end_list):
    text_span=[]
    if len(start_list)==0 or len(end_list)==0:
        return[]
    i=0
    j=0
    while i<len(start_list)and j<len(end_list):
        if start_list[i]<=end_list[j]:
            text_span.append([start_list[i],end_list[j]])
            i+=1
        else:
            j+=1
    return text_span

def build_span_labels(start_ids,end_ids):
    span_label_list=[]
    device=start_ids.device
    b=start_ids.size()[0]
    max_seq_len=start_ids.size()[1]
    for i in range(b):
        start_id_list=torch.nonzero(start_ids[i])
        end_id_list=torch.nonzero(end_ids[i])
        spans=get_span_for_eval(start_id_list,end_id_list)
        span_label = torch.zeros((max_seq_len, max_seq_len))

        for start, end in spans:
            span_label[start, end] = 1
        span_label_list.append(span_label)
    return torch.stack(span_label_list).to(device)

def epe_loss(start_ids, end_ids, span_logits, loss_fun):
    num_prompt=(span_logits > -1e10).sum().item()
    span_labels=build_span_labels(start_ids,end_ids)
    batch_size= span_logits.size()[0]
    span_labels = span_labels.reshape(batch_size , -1)
    span_logits = span_logits.reshape(batch_size , -1)
    loss = loss_fun(span_logits,span_labels)
    loss/=num_prompt

    return loss

class Text_encoder_with_epe(BertPreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        super(Text_encoder_with_epe, self).__init__(config)
        self.encoder = BertModel(config)
        self.config = config
        hidden_size = self.config.hidden_size
        adapt_span_params={'adapt_span_enabled':True,'adapt_span_loss':0.0,'adapt_span_ramp':32,'adapt_span_init':0.0,'adapt_span_cache':False}
        self.sparse_attn_layer=FSA_layer(hidden_size=hidden_size, nb_heads=8, attn_span=30, dropout=0.1, inner_hidden_size=hidden_size, adapt_span_params=adapt_span_params)
        self.sigmoid = nn.Sigmoid()
        self.BCE_loss=nn.BCEWithLogitsLoss(reduction="sum")
        self.use_vis=True
        self.epe=EPE(hidden_size=hidden_size)

        self.post_init()


    def forward(self, input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                prompt_mask: Optional[torch.Tensor] = None,
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict

        )
        sequence_output = outputs.last_hidden_state
        attentions=outputs.attentions
        if output_attentions:
            sequence_output,fuzzy_span_attentions = self.sparse_attn_layer(sequence_output,output_attentions=output_attentions)
        else:
            sequence_output = self.sparse_attn_layer(sequence_output,output_attentions=output_attentions)
            fuzzy_span_attentions=None

       
        span_logits=self.epe(sequence_output, attention_mask)
        span_logits=span_logits.squeeze(dim=1)
        span_logits=span_logits*prompt_mask-(1-prompt_mask)*1e12

        if start_positions is not None and end_positions is not None:
            loss_epe=epe_loss(start_ids=start_positions, end_ids=end_positions,
                             span_logits=span_logits, loss_fun=self.BCE_loss)
        else:
            loss_epe=None

        span_prob=self.sigmoid(span_logits)
        # pdb.set_trace()
        return UIEModelOutput(
            span_prob=span_prob,
            loss_epe=loss_epe,
            attentions=attentions,
            fuzzy_span_attentions=fuzzy_span_attentions,
            span_logits=span_logits
        )