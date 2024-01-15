# DQPSA
Code of paper "A Novel Energy Based Model Mechanism for Multi-Modal Aspect-Based Sentiment Analysis" in AAAI2024

### Prepare the environment
```
pip install -r requirements.txt
```

### Downloading pretrained model ckps and image features of datasets
```
link：https://pan.baidu.com/s/1bSf3OfEOWrRkd52UDHkokA 
Extracted code：2024
put the dir "data/" and "checkpoints/" under DQPSA/
put the dir "model_best/" under DQPSA/Text_encoder/
```

### Start training 
- use ```accelerate config --config_file deepspeed_ddp.json``` to create accelerate config fiting for your own device.
1. Training MATE model with [`train_MATE.sh`]
```
accelerate launch --config_file deepspeed_ddp.json MATE_finetune.py \
    --base_model ./Text_encoder/model_best \
    --pretrain_model ./checkpoints/pretrain_ckp/MASC_best_model.pt \
    --train_ds ./data/Twitter2015/MATE/train \
    --eval_ds ./data/Twitter2015/MATE/dev \
    --lr 2e-5 \
    --seed 1000 \
    --itc 1.0 \
    --itm 1.0 \
    --epe 1.0 \
    --save_path ./checkpoints/MATE_2015 \
    --epoch 20 \
    --log_step 1 \
    --save_step 1000 \
    --batch_size 6 \
    --accumulation_steps 2 \
    --val_step 200
```

2. Training MATE model with [`train_MASC.sh`]
```
accelerate launch --config_file deepspeed_ddp.json MASC_finetune.py \
    --base_model ./Text_encoder/model_best \
    --pretrain_model ./checkpoints/pretrain_ckp/MASC_best_model.pt \
    --train_ds ./data/Twitter2015/MASC/train \
    --eval_ds ./data/Twitter2015/MASC/dev \
    --lr 2e-5 \
    --seed 1000 \
    --itc 1.0 \
    --itm 1.0 \
    --epe 1.0 \
    --save_path ./checkpoints/MASC_2015 \
    --epoch 20 \
    --log_step 1 \
    --save_step 1000 \
    --batch_size 6 \
    --accumulation_steps 2 \
    --val_step 200
```

### Evaluating

- [`eval.sh`] has evaluation commands including MATE, MASC, and MABSA.

MATE (Twitter2015 as example)
```
python eval_tools.py \
    --MATE_model ./checkpoints/MATE_2015/best_f1:87.737.pt \
    --test_ds ./data/Twitter2015/MATE/test \
    --task MATE \
    --limit 0.5 \
    --device cuda:0
```

MASC (Twitter2015 as example)
```
python eval_tools.py \
    --MASC_model ./checkpoints/MASC_2015/best_f1:81.125.pt \
    --test_ds ./data/Twitter2015/MASC/test \
    --task MASC \
    --limit 0.5 \
    --device cuda:0
```

MABSA (Twitter2015 as example)
```
python eval_tools.py \
    --MATE_model ./checkpoints/MATE_2015/best_f1:87.737.pt \
    --MASC_model ./checkpoints/MASC_2015/best_f1:81.125.pt \
    --test_ds ./data/Twitter2015/MABSA/test \
    --task MABSA \
    --limit 0.5 \
    --device cuda:0
```
