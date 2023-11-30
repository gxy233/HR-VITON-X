python3 train_condition_distill.py \
--cuda True \
--gpu_ids 0 \
--name exp1 \
--keep_step 30000 \
--Ddownx2 \
--Ddropout \
--lasttvonly \
--interflowloss \
--occlusion
