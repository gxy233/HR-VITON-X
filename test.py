import torch
import torch.nn as nn
from networks import ConditionGenerator
import argparse
from cp_dataset import CPDataset, CPDataLoader


def get_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    # parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    # parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")
    # parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    # parser.add_argument('--num_D', type=int, default=2, help='Generator ngf')
    # # Cuda availability
    # parser.add_argument('--cuda', default=False, help='cuda or cpu')
    # parser.add_argument("--output_nc", type=int, default=13)
    # parser.add_argument("--enc_type", type=str, default='vit')
    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of input label classes without unknown class')
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    opt = parser.parse_args()
    return opt


opt = get_opt()
dataset = CPDataset(opt)
train_loader = CPDataLoader(opt, dataset)
inputs = train_loader.next_batch()
with open('test.log', 'w') as f:
    f.write('IM_name:\n')
    f.write(str(inputs['im_name']))
    f.write('cloth_name:\n')
    f.write(str(inputs['c_name']))
# cg = ConditionGenerator(opt, input1_nc=4, input2_nc=16, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
# total_params = sum(p.numel() for p in cg.parameters())
# print(f"Total parameters: {total_params}")
# trainable_params = sum(p.numel() for p in cg.parameters() if p.requires_grad)
# print(f"Trainable parameters: {trainable_params}")

