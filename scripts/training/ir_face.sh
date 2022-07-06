# Note: if you're training with fewer than 8 gpus, you should increase the per-gpu batch size controlled by
# the --batch argument so total batch size is preserved. Default value of --batch is 5 (assumes 8 gpus for training
# for a total batch size of 40 across all gpus)


PYTHONWARNINGS="ignore" torchrun --nproc_per_node=8 train.py \
--ckpt ir_face --load_G_only --padding_mode border --tv_weight 2500 \
--vis_every 5000 --ckpt_every 10000 --iter 150000 --anneal_psi 15000  --loss_fn lpips --exp-name ir_face






# %run train.py \
# --ckpt ir_face --load_G_only --padding_mode border --tv_weight 2500 \
# --vis_every 5000 --ckpt_every 50000 --iter 1500000 --loss_fn lpips --exp-name ir_face




# %run train.py \
# --ckpt celeba --load_G_only --padding_mode border --tv_weight 2500 \
# --vis_every 5000 --ckpt_every 50000 --iter 1500000 --loss_fn lpips --exp-name celeba 