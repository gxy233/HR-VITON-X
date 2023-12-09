# GAN-based Virtual Try-on

## Dataset availability

The dataset we use can be downloaded here: https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0.

## Code usage

### Installation

Clone this repository:

```bash
git clone 
cd ./
```

Install PyTorch and other dependencies:

```
conda create -n {env_name} python=3.8
pip install -r requirements.txt
```

Please then unzip the data into `./data`

### Training

There are 2 stages of training: train the condition generator and train the image generator.

#### Original HR-VITON

train the condition generator: `./train_CG_hr-viton.sh`

train the image generator: `./train_IG_hr-viton.sh`

#### ViT-based method

train the condition generator: `./train_CG_vit.sh`

train the image generator: `./train_IG_vit`

#### Distilled method

train the condition generator: `./train_CG_distill.sh`

train the image generator: `./train_IG_distill.sh`

The weights are saved in `./checkpoints`

### Testing

In this stage, remeber to change the path to the weights in the `.sh` file.

#### Original HR-VITON

test the condition generator: `./test_CG.sh`

generate the image: `./test_Generate_img.sh`

calculate the metrics: `./test_IG.sh`

#### ViT-based method

test the condition generator: `./test_CG.sh`

generate the image: `./test_Generate_img.sh`

calculate the metrics: `./test_IG.sh`

#### Distilled method

test the condition generator: `./test_CG.sh`

generate the image: `./test_Generate_img.sh`

calculate the metrics: `./test_IG.sh`

#### Test inference time

```bash
cd util
```
`./run_test_infertime.sh`

## Acknowledge

Our code is adapted from [HR-VITON](https://github.com/sangyun884/HR-VITON), originally developed and maintained by Sangyun Lee and colleagues. We have made modifications to suit our specific needs, but the core HR-VITON framework laid the groundwork for our development. We appreciate the efforts of the original authors in creating a resource that has significantly contributed to our project.