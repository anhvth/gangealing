rm -r data
export PYTHONPATH="${PYTHONPATH}:${PWD}"
IMG_DIR="/data/DMS_Drowsiness/all_video_symlink/01b4ff015835af4a07dd4dadd3d4a6d7/images"

SIZE=256
MIN_RES=50

python prepare_data.py --path $IMG_DIR --out data/new_lmdb_data --pad center --size $SIZE --pattern "*.jpg" --n_worker 32 --max_images 200

torchrun --nproc_per_node=6 applications/flow_scores.py --ckpt ir_face \
--real_data_path  data/new_lmdb_data --real_size $SIZE --no_flip_inference


torchrun --nproc_per_node=1 applications/congeal_dataset.py --ckpt pretrained/ir_face.pt \
    --real_data_path  data/new_lmdb_data \
    --out data/new_goli_align --real_size $SIZE --flow_scores  data/new_lmdb_data/flow_scores.pt \
    --fraction_retained 1 --output_resolution $SIZE --min_effective_resolution $MIN_RES


torchrun --nproc_per_node=1 applications/mixed_reality.py --ckpt celeba --objects \
    --label_path assets/objects/celeba/celeba_moustache.png --sigma 0.3 --opacity 1 --real_size 256 --resolution 4096 --real_data_path data/new_lmdb_data --no_flip_inference
