# python test_inference_time.py \
#     --modeltype gan1 \
#     --cuda True \
#     --gpu_ids 0 \
#     --datasetting paired \
#     --gen_checkpoint /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/image_generator/gen_model_final.pth \
#     --tocg_checkpoint /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/distilled_condition_generator/tocg_final.pth 

# python test_inference_time.py \
#     --modeltype ori \
#     --cuda True \
#     --gpu_ids 0 \
#     --datasetting paired \
#     --gen_checkpoint /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/image_generator/gen_model_final.pth \
#     --tocg_checkpoint /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/conditionG-tea/mtviton.pth

python test_inference_time.py \
    --modeltype gan2 \
    --cuda True \
    --gpu_ids 0 \
    --datasetting paired \
    --gen_checkpoint /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/image_G_dis/gen_step_040000.pth \
    --tocg_checkpoint /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/distilled_condition_generator/tocg_final.pth 

# python test_inference_time.py \
#     --modeltype vit \
#     --cuda True \
#     --gpu_ids 0 \
#     --datasetting paired \
#     --gen_checkpoint /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/image_generator/gen_model_final.pth \
#     --tocg_checkpoint /home/ubuntu/gxy/VITON-HD/HR-VITON/checkpoints/conditionG-tea/mtviton.pth