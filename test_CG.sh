# test Compressed gan model
# python test_condition.py  \
#     --gpu_ids 0 \
#     --modeltype compressed \
#     --tocg_checkpoint ./checkpoints/condition_generator_distilled/tocg_final.pth 


# test hr-viton model
python test_condition.py  \
    --gpu_ids 0 \
    --modeltype ori \
    --tocg_checkpoint ./checkpoints/conditionG-tea/mtviton.pth


# test vit model
# python test_condition.py  \
#     --gpu_ids 0 \
#     --modeltype vit \
#     --tocg_checkpoint ./checkpoints/condition_genrator_vit/tocg_step_200000.pth