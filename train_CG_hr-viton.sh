python3 train_condition.py \
    --cuda True \
    --gpu_ids 0 \
    --name hr-viton_CG \
    --keep_step 30000 \
    --Ddownx2 \
    --Ddropout \
    --lasttvonly \
    --interflowloss \
    --occlusion
