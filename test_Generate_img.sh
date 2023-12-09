# test compressed gan1
# python generate_image.py \
#     --modeltype gan1 \
#     --cuda True \
#     --gpu_ids 0 \
#     --test_name gan1_gen \
#     --output_dir ./output/gan1_gen_out \
#     --datasetting unpaired \
#     --gen_checkpoint ./image_generator/gen_model_final.pth \
#     --tocg_checkpoint ./checkpoints/condition_generator_distilled/tocg_final.pth \


# test compressed gan2
# python generate_image.py \
#     --modeltype gan2 \
#     --cuda True \
#     --gpu_ids 0 \
#     --test_name gan2_gen \
#     --output_dir ./output/gan2_gen_out \
#     --datasetting unpaired \
#     --gen_checkpoint ./checkpoints/image_generator_distill/gen_step_040000.pth \
#     --tocg_checkpoint ./checkpoints/condition_generator_distilled/tocg_final.pth \


# test vit
# python generate_image.py \
#     --modeltype vit \
#     --cuda True \
#     --gpu_ids 0 \
#     --test_name vit_gen \
#     --output_dir ./output/vit_gen_out \
#     --datasetting unpaired \
#     --gen_checkpoint ./checkpoints/image_generator/gen_model_final.pth \
#     --tocg_checkpoint ./checkpoints/image_generator_vit/gen_model_final.pth \


# test ori
python generate_image.py \
    --modeltype ori \
    --cuda True \
    --gpu_ids 0 \
    --test_name ori_gen \
    --output_dir ./output/ori_gen_out \
    --datasetting unpaired \
    --gen_checkpoint ./checkpoints/image_generator_tea/gen.pth \
    --tocg_checkpoint ./checkpoints/conditionG-tea/mtviton.pth \
