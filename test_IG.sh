# test compressed gan1
# python test_generator.py \
#     --modeltype gan1 \
#     --cuda True \
#     --gpu_ids 0 \
#     --test_name gan1_test \
#     --output_dir ./gan1_test_out \
#     --datasetting paired \
#     --enc_type res \
#     --gen_checkpoint ./image_generator/gen_model_final.pth \
#     --metric_freq 50 \
#     --tocg_checkpoint ./checkpoints/condition_generator_distilled/tocg_final.pth \
#     --save_img False

# test compressed gan2
# python test_generator.py \
#     --modeltype gan2 \
#     --cuda True \
#     --gpu_ids 0 \
#     --test_name gan2_test \
#     --output_dir ./gan1_test_out \
#     --datasetting paired \
#     --enc_type res \
#     --gen_checkpoint ./checkpoints/image_generator_tea/gen_step_040000.pth \
#     --metric_freq 50 \
#     --tocg_checkpoint ./checkpoints/condition_generator_distilled/tocg_final.pth \
#     --save_img False

# test vit
python test_generator.py \
    --modeltype vit \
    --cuda True \
    --gpu_ids 0 \
    --test_name vit_test \
    --output_dir ./vit_test_out \
    --datasetting paired \
    --enc_type vit \
    --gen_checkpoint ./checkpoints/image_generator/gen_model_final.pth \
    --metric_freq 50 \
    --tocg_checkpoint ./checkpoints/image_generator_vit/gen_model_final.pth \
    --save_img False

# test ori
# python test_generator.py \
#     --modeltype ori \
#     --cuda True \
#     --gpu_ids 0 \
#     --test_name ori_test \
#     --output_dir ./ori_test_out \
#     --datasetting paired \
#     --enc_type res \
#     --gen_checkpoint ./checkpoints/image_generator/gen_model_final.pth \
#     --metric_freq 50 \
#     --tocg_checkpoint ./checkpoints/conditionG-tea/mtviton.pth \
#     --save_img False
