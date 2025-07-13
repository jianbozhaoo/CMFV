# fever_2class
CUDA_VISIBLE_DEVICES="0" python train_fever_2class.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128 

# politihop_3class
CUDA_VISIBLE_DEVICES="0" python train_politihop_3class.py \
--seed 1234 \
--batch_size 4 \
--lr 2e-6 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 

# politihop_2class
CUDA_VISIBLE_DEVICES="0" python train_politihop_2class.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 10 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 