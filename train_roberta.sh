
python3 train_xlm_roberta.py --mode train \
--lr 0.00001 \
--batch_size 16 \
--epoch 5 \
--max_len 100 \
--date 20220923 \
--seed 42 \
--gpu 1
# --wandb