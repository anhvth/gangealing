# python prepare_data.py --out datasets/ir_lmdb --n_worker 16 --size 256 datasets/ir



python -m torch.distributed.launch --nproc_per_node=1 --master_port=33221 train.py --batch 16 datasets/ir_lmdb --ckpt pretrained/celeba.pt