python3  train_generator_distill.py \
--cuda True \
--name image_G_dis \
-b 1 \
-j 8 \
--gpu_ids 0 \
--tocg_checkpoint  /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/distilled_condition_generator/tocg_final.pth \
--Tdis_checkpoint ./checkpoints/image_discriminator_tea/D_step_260000.pth \
--Tgen_checkpoint ./checkpoints/image_generator_tea/gen.pth \
--occlusion