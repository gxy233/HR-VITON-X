python3  train_generator.py \
--cuda True \
--name hr-viton_IG \
-b 1 \
-j 8 \
--gpu_ids 0 \
--keep_step 60000 \
--decay_step 100000 \
--tocg_checkpoint  ./checkpoints/conditionG-tea/mtviton.pth \
--occlusion
