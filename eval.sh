# MATE evaluation
python eval_tools.py \
    --MATE_model ./checkpoints/MATE_2015/best_f1:87.737.pt \
    --test_ds ./data/Twitter2015/MATE/test \
    --task MATE \
    --limit 0.5 \
    --device cuda:0

# MASC evaluation
#python eval_tools.py \
#    --MASC_model ./checkpoints/MASC_2015/best_f1:81.125.pt \
#    --test_ds ./data/Twitter2015/MASC/test \
#    --task MASC \
#    --limit 0.5 \
#    --device cuda:0

# MABSA evaluation
#python eval_tools.py \
#    --MATE_model ./checkpoints/MATE_2015/best_f1:87.737.pt \
#    --MASC_model ./checkpoints/MASC_2015/best_f1:81.125.pt \
#    --test_ds ./data/Twitter2015/MABSA/test \
#    --task MABSA \
#    --limit 0.5 \
#    --device cuda:0