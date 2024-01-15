import torch
import argparse
import better_exceptions
import os
import random
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from model import from_pretrained
import numpy as np
import torch
from torch.utils.data import DataLoader
from DQPSA_dataset import collate_fn, DQPSA_dataset
# import pdb
import logging
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from eval_tools import *




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def save_checkpoint(state, save_path, step):
    # 创建当前的存储目录
    os.makedirs(save_path, exist_ok=True)
    if isinstance(step, str):
        save_dir=os.path.join(save_path,f"{step}.pt")
    else:
        save_dir=os.path.join(save_path,f"step_{step/1000}k.pt")
    # 存储模型和训练状态
    accelerator.save(state, save_dir)


def compute_metric(total_correct,total_label,total_pred):
    precision = total_correct / total_pred if total_correct else 0.0
    recall=total_correct/total_label if total_correct else 0.0
    f1=(2 * (precision * recall) / (precision + recall)) if total_correct else 0.0
    return precision,recall,f1
    
def finetune(accelerator,model, optimizer,  train_dataset, num_epoch, log_step,save_step,val_step,batch_size,save_path,device,accumulation_steps,scheduler=None,eval_dataset=None):

    torch.autograd.set_detect_anomaly(True)

    os.makedirs(os.path.join(save_path,"log_path"), exist_ok=True)
    writer = SummaryWriter(os.path.join(save_path,"log_path"))
    epoch=0
    step=0
    sum_loss=0.0
    best_f1=0.0



    print("loading test set")
    eval_dataset.update_data()
    eval_dataloader=DataLoader(eval_dataset,batch_size = batch_size,collate_fn=collate_fn,shuffle=True)
    accelerator.wait_for_everyone()
    print("loading train set")
    model, optimizer,scheduler,eval_dataloader = accelerator.prepare(model, optimizer,scheduler,eval_dataloader)
    model.to(device)
    accelerator.wait_for_everyone()
    model.train()
    for epoch_index in range(epoch,num_epoch):
        train_dataset.reset()
        dataset_num=0
        while not train_dataset.is_end():
            dataset_num+=1
            train_dataset.update_data()
            train_dataloader=DataLoader(train_dataset,batch_size = batch_size,collate_fn=collate_fn,shuffle=True)
            train_dataloader = accelerator.prepare(train_dataloader)

            for batch in tqdm(train_dataloader, desc=f"precess:{accelerator.state.process_index},epoch:{epoch_index},dataset_num:{dataset_num}"):
                torch.cuda.empty_cache()
                batch["image_embeds"]=batch["image_embeds"].to(device)
                batch["query_inputs"] = batch["query_inputs"].to(device)
                batch["answer_inputs"]['input_ids'] = batch["answer_inputs"]['input_ids'].to(device)
                batch["answer_inputs"]['attention_mask'] = batch["answer_inputs"]['attention_mask'].to(device)
                batch["IE_inputs"]['input_ids'] = batch["IE_inputs"]['input_ids'].to(device)
                batch["IE_inputs"]['attention_mask'] = batch["IE_inputs"]['attention_mask'].to(device)
                batch["start_ids"]=batch["start_ids"].to(device)
                batch["end_ids"]=batch["end_ids"].to(device)
                batch["prompt_mask"]=batch["prompt_mask"].to(device)
                step+=1
                with maybe_autocast(model=model):
                    model_output=model(batch)
                    accelerator.wait_for_everyone()
                    loss = model_output.total_loss

                average_loss = accelerator.gather(model_output.total_loss)
                sum_loss+=average_loss.mean().item()
                avg_loss=sum_loss/step

                if accelerator.is_local_main_process:
                    print(average_loss.mean().item())

                accelerator.wait_for_everyone()

                if step%val_step==0:
                    accelerator.wait_for_everyone()
                    limit=0.1
                    del batch
                    torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()
                    eval_res=eval_MATE(model,eval_dataloader,limit,device)
                    accelerator.wait_for_everyone()

                    total_correct=accelerator.gather(eval_res[0]).sum()
                    total_label=accelerator.gather(eval_res[1]).sum()
                    total_pred=accelerator.gather(eval_res[2]).sum()
                    accelerator.wait_for_everyone()
                    total_precision,total_recall,total_f1=compute_metric(total_correct.item(),total_label.item(),total_pred.item())

                    if accelerator.is_local_main_process:
                        print("total_correct:",total_correct)
                        print("total_label:",total_label)
                        print("total_pred:",total_pred)
                        print("P:",total_precision)
                        print("R:",total_recall)
                        print("f1:",total_f1)
                        writer.add_scalar('evaluate/total_P', total_precision, step)
                        writer.add_scalar('evaluate/total_R', total_recall, step)
                        writer.add_scalar('evaluate/total_F1', total_f1, step)
                    
                    accelerator.wait_for_everyone()
                    if total_f1>best_f1 and accelerator.is_local_main_process:
                        best_f1=total_f1
                        unwrapped_model = accelerator.unwrap_model(model)
                        # save
                        save_checkpoint(unwrapped_model.state_dict(), save_path, f"best_f1:{format(100*total_f1, '.3f')}")
                accelerator.wait_for_everyone()

                if step%log_step==0 and accelerator.is_local_main_process:
                    writer.add_scalar('step_loss/total_loss', average_loss.mean(), step)
                    writer.add_scalar('step_loss/avg_loss', avg_loss, step)

                accelerator.wait_for_everyone()
                accelerator.backward(loss)
                accelerator.wait_for_everyone()

                if step%accumulation_steps==0:
                    optimizer.step()
                    optimizer.zero_grad()
                    model.module.text_encoder.sparse_attn_layer.attn.attn.adaptive_span.clamp_param()
                
                accelerator.wait_for_everyone()

                if  step%save_step==0 and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    # save
                    save_checkpoint(unwrapped_model.state_dict(), save_path, step)
                    logging.info(f"checkpoint saved in {save_path}")
                    writer.add_scalar('checkpoint/total_loss', average_loss.mean(), step)
                    writer.add_scalar('checkpoint/avg_loss', avg_loss, step)
                
                accelerator.wait_for_everyone()
        if scheduler is not None:
            scheduler.step()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    accelerator = Accelerator()
    device=accelerator.device

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="./Text_encoder/model_best", )
    parser.add_argument('--pretrain_model', type=str, default="./Text_encoder/model_best", )
    parser.add_argument('--train_ds', type=str, default="./playground/twitter2015/MATE/train")
    parser.add_argument('--eval_ds', type=str, default="./playground/twitter2015/MATE/dev")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--itc', type=float, default=1.0)
    parser.add_argument('--itm', type=float, default=1.0)
    parser.add_argument('--epe', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--save_step', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=200)
    parser.add_argument('--save_path', type=str, default="./checkpoints/MATE_2015")
    args = parser.parse_args()

    #build_dataset
    PQ_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    IE_tokenizer=BertTokenizer.from_pretrained(args.base_model)

    train_ds= DQPSA_dataset(
                    data_path=args.train_ds,
                    max_seq_len=512,
                    IE_tokenizer=IE_tokenizer,
                    PQ_former_tokenizer=PQ_tokenizer,
                    num_query_token=32,
                    SEP_token_id=2,
                    split_token_id=187284,
                    set_size=3)
    eval_ds= DQPSA_dataset(
                    data_path=args.eval_ds,
                    max_seq_len=512,
                    IE_tokenizer=IE_tokenizer,
                    PQ_former_tokenizer=PQ_tokenizer,
                    num_query_token=32,
                    SEP_token_id=2,
                    split_token_id=187284,
                    set_size=3)

    set_seed(args.seed)
    model=from_pretrained(args.pretrain_model)

    model.itc_weight=args.itc
    model.itm_weight=args.itm
    model.epe_weight=args.epe

    optimizer = AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, betas=(0.9, 0.98), weight_decay=0.05)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2*1000, eta_min=5e-5) 
    print("start training")
    finetune(
        model=model,
        optimizer=optimizer, 
        train_dataset=train_ds, 
        num_epoch=args.epoch, 
        log_step=args.log_step,
        save_step=args.save_step,
        batch_size=args.batch_size,
        save_path=args.save_path,
        accelerator=accelerator,
        device=device,
        scheduler=scheduler,
        accumulation_steps=args.accumulation_steps,
        eval_dataset=eval_ds,
        val_step=args.val_step
        )

#accelerate launch --config_file deepspeed_ddp.json MATE_finetune.py



