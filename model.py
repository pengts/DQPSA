import sys
sys.path.append('Text_encoder')
sys.path.append('PDQ')
from transformers import BertTokenizerFast
from Text_encoder.sparse_attn_model import Text_encoder_with_epe
from PDQ.PDQ import PDQ
from transformers.utils import ModelOutput
from typing import Optional, Tuple
import torch.nn as nn
import torch

class DQPSA_Output(ModelOutput):
    total_loss: Optional[torch.FloatTensor] = None
    loss_bce: Optional[torch.FloatTensor] = None
    loss_fsl: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_epe: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None


def get_pred_span(start_ids, end_ids):
    start_list = torch.nonzero(start_ids)
    end_list = torch.nonzero(end_ids)
    start_list = [x[0] - 32 for x in start_list]
    end_list = [x[0] - 32 + 1 for x in end_list]
    text_span = []
    if len(start_list) == 0 or len(end_list) == 0:
        return []
    i = 0
    j = 0
    while i < len(start_list) and j < len(end_list):
        if start_list[i] < end_list[j]:
            text_span.append([start_list[i], start_list[i]])
            i += 1
        else:
            j += 1
    return text_span


def build_tokenizer(tokenizer_path):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


class DQPSA(nn.Module):
    def __init__(self, MFSUIE_config):
        super().__init__()
        self.pdq = PDQ()
        self.text_encoder = Text_encoder_with_epe.from_pretrained(MFSUIE_config['text_model']["model_path"])
        self.tokenizer = build_tokenizer(MFSUIE_config["text_model"]["tokenizer_path"])
        Qformer_hidden_size = MFSUIE_config["pdq"]["hidden_size"]
        text_hidden_size = MFSUIE_config["text_model"]["hidden_size"]
        self.FSUIE_proj = nn.Linear(Qformer_hidden_size, text_hidden_size)
        self.itc_weight = MFSUIE_config["loss_weights"]["itc"]
        self.itm_weight = MFSUIE_config["loss_weights"]["itm"]
        self.epe_weight = MFSUIE_config["loss_weights"]["epe"]
        self.dropout_layer = nn.Dropout()

    def forward(self, samples, no_its_and_itm=False):
        PQformer_outputs = self.pdq(samples, no_its_and_itm)
        query_outputs = self.FSUIE_proj(PQformer_outputs.FSUIE_inputs)
        query_outputs = self.dropout_layer(query_outputs)
        text_attn = torch.ones(query_outputs.size()[:-1], dtype=torch.long).to(query_outputs.device)
        text_input_ids = samples['IE_inputs']['input_ids']
        text_att_mask = samples['IE_inputs']['attention_mask']
        start_ids = samples["start_ids"]
        end_ids = samples["end_ids"]
        prompt_mask = samples["prompt_mask"]
        text_encoder_atts = torch.cat([text_attn, text_att_mask], dim=1)
        text_inputs_embeds = self.text_encoder.encoder.embeddings(input_ids=text_input_ids)
        text_inputs_embeds = torch.cat([query_outputs, text_inputs_embeds], dim=1)
        FSUIE_outputs = self.text_encoder(inputs_embeds=text_inputs_embeds,
                                          # token_type_ids=token_type_ids,
                                          attention_mask=text_encoder_atts,
                                          start_positions=start_ids,
                                          end_positions=end_ids,
                                          prompt_mask=prompt_mask)
        # FSUIE_outputs: #loss,loss_bce,loss_fsl,start_prob,end_prob
        total_loss = (self.itc_weight * PQformer_outputs.loss_itc
                      + self.itm_weight * PQformer_outputs.loss_itm
                      + self.epe_weight * FSUIE_outputs.loss_epe
                      )
        return DQPSA_Output(
            total_loss=total_loss,
            loss_epe=FSUIE_outputs.loss_epe,
            loss_itm=PQformer_outputs.loss_itm,
            loss_itc=PQformer_outputs.loss_itc,
            span_prob=FSUIE_outputs.span_prob,
            span_logits=FSUIE_outputs.span_logits
        )


def from_pretrained(path):
    pretrain_config = {
        "text_model": {"model_path": "./Text_encoder/model_best",
                       "tokenizer_path": "./Text_encoder/model_best",
                       "hidden_size": 768
                       },
        "pdq": {
            "hidden_size": 768
        },
        "loss_weights": {"itc": 1.0, "itm": 1.0, "epe": 1.0},
        "rand_seed": 0,
        "lr": 5e-5
    }
    model = DQPSA(pretrain_config)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    print(f"loading model finished from {path}")
    return model
