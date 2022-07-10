# Note: if you're training with fewer than 8 gpus, you should increase the per-gpu batch size controlled by
# the --batch argument so total batch size is preserved. Default value of --batch is 5 (assumes 8 gpus for training
# for a total batch size of 40 across all gpus)


# PYTHONWARNINGS="ignore" torchrun --nproc_per_node=8 train.py  \
# --ckpt ir_face --padding_mode border --tv_weight 2500 \
# --vis_every 5000 --ckpt_every 10000 --anneal_psi 15000 --iter 80000 --period 3750 \
#  --loss_fn lpips --load_G_only \
#  --exp-name ir_face --batch 5 --num_heads 3



 # Note: if you're training with fewer than 8 gpus, you should increase the per-gpu batch size controlled by
# the --batch argument so total batch size is preserved. Default value of --batch is 5 (assumes 8 gpus for training
# for a total batch size of 40 across all gpus)


python prepare_data.py --path ./data/generated_ir_face --out data/generated_ir_face_lmdb --pad border --size 256 --pattern "*.jpg"



PYTHONWARNINGS="ignore"  torchrun --nproc_per_node=8 train_cluster_classifier.py \
--ckpt results/ir_face/checkpoints/0080000.pt --padding_mode border \
--vis_every 500 --ckpt_every 5000 --iter 55000 --period 50000 --loss_fn lpips --exp-name ir_face_cluster_classifier \
--num_heads 3  --ndirs 1 --inject 5 --sample_from_full_res --real_data_path data/generated_ir_face_lmdb
