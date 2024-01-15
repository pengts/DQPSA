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