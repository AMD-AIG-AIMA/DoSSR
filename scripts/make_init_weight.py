import sys
sys.path.append(".")
from argparse import ArgumentParser
from typing import Dict
import os
import torch
from omegaconf import OmegaConf

from utils.common import instantiate_from_config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_weight(weight_path: str) -> Dict[str, torch.Tensor]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weight = torch.load(weight_path, map_location=device)
    if "state_dict" in weight:
        weight = weight["state_dict"]

    pure_weight = {}
    for key, val in weight.items():
        if key.startswith("module."):
            key = key[len("module."):]
        pure_weight[key] = val

    return pure_weight



def load_weight_esrgan(weight_path: str) -> Dict[str, torch.Tensor]:
    weight = torch.load(weight_path)['params_ema']
    if "state_dict" in weight:
        weight = weight["state_dict"]

    pure_weight = {}
    for key, val in weight.items():
        if key.startswith("module."):
            key = key[len("module."):]
        pure_weight[key] = val

    return pure_weight



parser = ArgumentParser()
parser.add_argument("--cldm_config", type=str, required=True)
parser.add_argument("--sd_weight", type=str, required=True)
parser.add_argument("--preprocess_model_weight", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

model = instantiate_from_config(OmegaConf.load(args.cldm_config))

sd_weights = load_weight(args.sd_weight)
preprocess_model_weight = load_weight_esrgan(args.preprocess_model_weight)
scratch_weights = model.state_dict()

# # print(sd_weights.keys())
# for key in sd_weights.keys():
#     print(str(key))
init_weights = {}
for weight_name in scratch_weights.keys():
    # find target pretrained weights for this weight
    if weight_name.startswith("preprocess_model."):
        suffix = weight_name[len("preprocess_model."):]
        target_name = suffix
        target_model_weights = preprocess_model_weight
    else:
        target_name = weight_name
        target_model_weights = sd_weights
    
    # if target weight exist in pretrained model
    print(f"copy weights: {target_name} -> {weight_name}")
    if target_name in target_model_weights:
        # get pretrained weight
        target_weight = target_model_weights[target_name]
        target_shape = target_weight.shape
        model_shape = scratch_weights[weight_name].shape
        # if pretrained weight has the same shape with model weight, we make a copy
        if model_shape == target_shape:
            init_weights[weight_name] = target_weight.clone()
        # else we copy pretrained weight with additional channels initialized to zero
        else:
            newly_added_channels = model_shape[1] - target_shape[1]
            oc, _, h, w = target_shape
            zero_weight = torch.zeros((oc, newly_added_channels, h, w)).type_as(target_weight)
            init_weights[weight_name] = torch.cat((target_weight.clone(), zero_weight), dim=1)
            print(f"add zero weight to {target_name} in pretrained weights, newly added channels = {newly_added_channels}")
    else:
        print("++++++++==============+++++++++++")
        init_weights[weight_name] = scratch_weights[weight_name].clone()
        print(f"These weights are newly added: {weight_name}")

model.load_state_dict(init_weights, strict=True)
torch.save(model.state_dict(), args.output)
print("Done.")
