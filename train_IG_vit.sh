python3 train_generator_vit.py \
    --cuda True \
    --name vit_gen_100k \
    --enc_type vit \
    --keep_step 50000 \
    --decay_step 50000 \
    --tocg_checkpoint ./checkpoints/condition_genrator_vit/tocg_step_200000.pth \
    -b 1 \
    -j 4 \
    --gpu_ids 0 \
    --occlusion