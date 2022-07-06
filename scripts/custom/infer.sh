# torchrun --nproc_per_node=1 applications/mixed_reality.py \
#     --ckpt celeba --objects --label_path assets/masks/celeba_mask.png \
#     --sigma 0.3 --opacity 1 --real_size 256 --resolution 1024 \
#     --real_data_path data/sample_croped_video --no_flip_inference


torchrun --nproc_per_node=1 applications/mixed_reality.py --ckpt celeba --objects \
    --label_path assets/objects/celeba/celeba_moustache.png --sigma 0.3 --opacity 1 --real_size 256 --resolution 4096 \
    --real_data_path data/sample_croped_video --no_flip_inference




torchrun --nproc_per_node=1 applications/congeal_dataset.py --ckpt ir_face \
--real_data_path data/sample_croped_video --out data/sample_croped_video_aligned_dataset --real_size 256 \
--flow_scores data/sample_croped_video_aligned_dataset/flow_scores.pt --fraction_retained 1 --output_resolution 256

torchrun --nproc_per_node=1 applications/congeal_dataset.py --ckpt ir_face --real_data_path data/sample_croped_video \
--out data/my_new_aligned_dataset --real_size 0 --flow_scores data/sample_croped_video_aligned_dataset/flow_scores.pt \
--fraction_retained 1 \
    --output_resolution 256