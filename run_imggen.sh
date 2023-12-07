python3  train_generator_distill.py \
--cuda True \
--name image_G_dis \
-b 1 \
-j 8 \
--gpu_ids 0 \
--keep_step 60000 \
--decay_step 100000 \
--tocg_checkpoint  /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/distilled_condition_generator/tocg_final.pth \
--Tdis_checkpoint ./checkpoints/image_discriminator_tea/D_step_260000.pth \
--Tgen_checkpoint ./checkpoints/image_generator_tea/gen.pth \
--Sgen_checkpoint ./checkpoints/image_G_dis/gen_step_040000.pth \
--occlusion