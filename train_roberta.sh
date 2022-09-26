
python3 train_xlm_roberta.py --mode predict \
--lr 0.00001 \
--batch_size 16 \
--epoch 5 \
--max_len 300 \
--date 20220925 \
--seed 42 \
--gpu 1
# --wandb