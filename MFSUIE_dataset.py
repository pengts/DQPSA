
import torch
import sys
sys.path.append('Text_encoder')
sys.path.append('PDQ')
from torch.utils.data import Dataset, DataLoader
import logging
import os
# import skimage
# import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import open_clip
from collections import OrderedDict
import torch
from transformers import BertTokenizer
import pdb
from tqdm import tqdm
import os
import pickle
import random
import json

def get_tgt(target,tokenizer):
    tgt_list = [tokenizer.encode(x, add_special_tokens=False) for x in target]
    return tgt_list


def get_span(target,input_ids,tokenizer):
    # 假设已有tokenizer和input_ids

    # 将待搜索字符串转换为token并获取其长度
    tgt_tokens = tokenizer.encode(target, add_special_tokens=False)
    ###
    # tgt_tokens=tokenizer.convert_ids_to_tokens(tgt_tokens)
    # tgt_tokens[0]=tgt_tokens[0][1:]
    # tgt_tokens=tokenizer.convert_tokens_to_ids(tgt_tokens)
    ###
    tgt_token_len = len(tgt_tokens)
    input_ids_list=input_ids.tolist()
    # 在input_ids中查找待搜索字符串的起始和结束位置
    start_pos = []
    end_pos = []
    for i in range(len(input_ids_list)-tgt_token_len+1):
        if input_ids_list[i:i+tgt_token_len] == tgt_tokens:
            start_pos.append(i)
            end_pos.append(i + tgt_token_len - 1)
    # pdb.set_trace()
    return start_pos,end_pos



class MFSUIE_dataset(Dataset):

    def __init__(self,
                IE_tokenizer,
                PQ_former_tokenizer,
                data_path,                
                max_seq_len=512,
                num_query_token=32,
                SEP_token_id=2,
                split_token_id=187284,
                set_size=10,
                with_label=False,
                with_prompt_mask=True
                ):
        super().__init__()
        #init data
        self.data=[]
        filelist = os.listdir(data_path)
        data_filelist=[x for x in filelist if x.endswith("pkl")]
        self.data_path=[os.path.join(data_path,fl) for fl in data_filelist]

        label_filelist=[x for x in filelist if x.endswith("json")]
        label_filelist=[os.path.join(data_path,fl) for fl in label_filelist]

        # random.shuffle(self.data_path) 
        print(len(self.data_path))
        self.set_size=set_size
        #other obj
        self.max_seq_len = max_seq_len
        self.num_query_token=num_query_token
        self.PQ_former_tokenizer=PQ_former_tokenizer
        self.IE_tokenizer=IE_tokenizer
        self.SEP_token_id=SEP_token_id
        self.split_token_id=split_token_id
        self.current_data_index=0
        self.with_label=with_label
        self.with_prompt_mask=with_prompt_mask
        if with_label:
            self.label_data=[]
            for x in label_filelist:
                with open (x,"r")as f:
                    temp=json.load(f)
                    self.label_data.extend(temp)
        else:
            self.label_data=None
            

    def update_data(self):
        set_size=self.set_size
        start_idx=self.current_data_index
        end_idx=start_idx+set_size if start_idx+set_size<len(self.data_path)+1 else len(self.data_path)
        current_data=self.data_path[start_idx:end_idx]
        self.data=[]
        for path in tqdm(current_data,desc="data loading"):
            with open(path, 'rb') as f:
                temp=pickle.load(f)
                self.data.extend(temp)
        self.current_data_index=end_idx
        print("index here:",self.current_data_index)
    
    def is_end(self):
        return self.current_data_index==len(self.data_path)

    def reset(self):
        self.current_data_index=0

    def __getitem__(self, index):

        image_feature=torch.from_numpy(self.data[index]["image_feature"])#.half()


        query_inputs = self.PQ_former_tokenizer(
            self.data[index]["query_input"],
            padding="max_length",
            truncation=True,
            max_length=self.num_query_token,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]

        # pdb.set_trace()

        answer_inputs = self.PQ_former_tokenizer(
            self.data[index]["answer_input"],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        answer_inputs={
                "input_ids":answer_inputs["input_ids"],
                "attention_mask":answer_inputs["attention_mask"]
        }


        IE_inputs = self.IE_tokenizer(
            text=self.data[index]["query_input"],
            text_pair=self.data[index]["text_input"].replace(" ###",","),
            padding="max_length",
            truncation=True,
            max_length=(self.max_seq_len-self.num_query_token),
            add_special_tokens=True,
            #return_offsets_mapping=True
        )
        # pdb.set_trace()
        IE_inputs["input_ids"]=[self.SEP_token_id if x == self.split_token_id else x for x in IE_inputs["input_ids"]]
        
        IE_inputs["input_ids"]=torch.tensor(IE_inputs["input_ids"]).int()
        IE_inputs["attention_mask"]=torch.tensor(IE_inputs["attention_mask"]).int()
        
        IE_inputs={
                    "input_ids":IE_inputs["input_ids"],
                    "attention_mask":IE_inputs["attention_mask"]
        }

        start_ids = torch.zeros(self.max_seq_len).int()
        end_ids = torch.zeros(self.max_seq_len).int()
        if isinstance(self.data[index]["target"],list):
            for i in self.data[index]["target"]:
                start_pos_list,end_pos_list=get_span(target=i,
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=self.IE_tokenizer)
                
                # for start_pos in start_pos_list:
                #     start_ids[start_pos+self.num_query_token] = 1
                # for end_pos in end_pos_list:
                #     end_ids[end_pos+self.num_query_token] = 1

                for i in range(len(start_pos_list)):
                    if (not start_ids[start_pos_list[i]+self.num_query_token]==1) and (not end_ids[end_pos_list[i]+self.num_query_token]==1):
                        start_ids[start_pos_list[i]+self.num_query_token]=1
                        end_ids[end_pos_list[i]+self.num_query_token] = 1
        else:
            start_pos_list,end_pos_list=get_span(self.data[index]["target"],
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=self.IE_tokenizer)
            for start_pos in start_pos_list:
                start_ids[start_pos+self.num_query_token] = 1
            for end_pos in end_pos_list:
                end_ids[end_pos+self.num_query_token] = 1

        # target_list=get_tgt(target=self.data[index]["target"],
        #                             tokenizer=self.IE_tokenizer)

        #attention: obj in answer_inputs is [1,512], in IE_inputs is [512]
        
        # return (image_feature, query_inputs, answer_inputs, IE_inputs, start_ids, end_ids,self.data[index]["target"])#,target_list)
        res=[image_feature, query_inputs, answer_inputs, IE_inputs, start_ids, end_ids]

        if self.with_label:
            res.append(self.label_data[index])
        else:
            res.append(None)

        if self.with_prompt_mask:
            if "[ positive, neutral, negative ]" in self.data[index]["query_input"]:
                prompt_target="[ positive, neutral, negative ]"
            else:
                prompt_target=self.data[index]["text_input"]

            prompt_mask = torch.zeros(self.max_seq_len,self.max_seq_len).int()
            prompt_start_list,prompt_end_list=get_span(prompt_target,
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=self.IE_tokenizer)
            for start_pos,end_pos in zip(prompt_start_list,prompt_end_list):
                prompt_mask[start_pos+self.num_query_token:end_pos+self.num_query_token+1,
                            start_pos+self.num_query_token:end_pos+self.num_query_token+1]=1
            res.append(prompt_mask)
        else:
            res.append(None)
        
        return tuple(res)

    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
    #batch:[image_feature, query_inputs, answer_inputs, IE_inputs, start_ids, end_ids]
    image_embeds=torch.stack([b[0] for b in batch], dim=0)
    query_inputs=torch.stack([b[1] for b in batch], dim=0)
    # answer_inputs_input_ids=torch.stack([b[2]["input_ids"][0] for b in batch], dim=0)
    # answer_inputs_attention_mask=torch.stack([b[2]["attention_mask"][0] for b in batch], dim=0)
    # IE_inputs_input_ids=torch.stack([b[3]["input_ids"] for b in batch], dim=0)
    # IE_inputs_attention_mask=torch.stack([b[3]["attention_mask"] for b in batch], dim=0)
    answer_inputs={
                    "input_ids":torch.stack([b[2]["input_ids"][0] for b in batch], dim=0),
                    "attention_mask":torch.stack([b[2]["attention_mask"][0] for b in batch], dim=0)
                    }
    IE_inputs={
                "input_ids":torch.stack([b[3]["input_ids"] for b in batch], dim=0),
                "attention_mask":torch.stack([b[3]["attention_mask"] for b in batch], dim=0)
                    }
    start_ids=torch.stack([b[4] for b in batch], dim=0)
    end_ids=torch.stack([b[5] for b in batch], dim=0)

    sample={"image_embeds":image_embeds,"query_inputs":query_inputs,
                "answer_inputs":answer_inputs,"IE_inputs":IE_inputs,
                "start_ids":start_ids,"end_ids":end_ids}
    try:
        if batch[0][6]!=None:
            label_data=[x[6] for x in batch ]
            sample["label_data"]=label_data
    except:
        pdb.set_trace()


    if batch[0][7]!=None:
        prompt_mask=torch.stack([b[7] for b in batch], dim=0)
        sample["prompt_mask"]=prompt_mask
    
    return sample

if __name__=="__main__":
    PQ_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    IE_tokenizer=BertTokenizer.from_pretrained('Text_encoder/FSUIE_tokenizer')
    eval_ds= MFSUIE_dataset( 
                    data_path="/data1/pengts/Twitter15/processed_Twitter15/test",
                    max_seq_len=512,
                    IE_tokenizer=IE_tokenizer,
                    PQ_former_tokenizer=PQ_tokenizer,
                    num_query_token=32,
                    SEP_token_id=2,
                    split_token_id=187284,
                    set_size=1)
    eval_ds.update_data()
    eval_dataloader=DataLoader(eval_ds,batch_size =3,collate_fn=collate_fn,shuffle=False)
    batch=next(iter(eval_dataloader))
    input_id=batch["IE_inputs"]["input_ids"][1]
    prompt_mask=batch["prompt_mask"][1]
    print(input_id)
    print(prompt_mask)
    pdb.set_trace()
