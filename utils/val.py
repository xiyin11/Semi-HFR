'''
    implement the feature extractions for light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

from __future__ import print_function
import imp
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


from PIL import Image
import logging
import numpy as np
import cv2
import sys
import heapq
import time

sys.path.append('.')
from model.lightcnn_v4_MCR import LightCNN_V4
from utils.init import get_config,argument_parser,setup
from utils.datasets.LAMP_HQ import LAMP_HQ
from utils.datasets.NIR_VISv2 import NIR_VISv2
from utils.datasets.Tufts import Tufts

logger=logging.getLogger("Train")
    

def main(args):
    cfg=get_config(args.config_file)
    setup(cfg,args)
    model = LightCNN_V4().cuda()

    model.eval()
    checkpoint = torch.load(cfg.WEIGHTS)
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc2' not in k and 'middle' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    val_cfg=get_config(args.config_file)

    val_model(model,val_cfg)

def val_model(model,cfg):
    model.eval()
    # val_cfg=get_config(config)
    probe_list=[]
    gallery_list=[]
    pid_dict={}
    eval(cfg.DATASET).get_list(gallery_list,os.path.join(cfg.ROOT_PATH,cfg.GALLERY),pid_dict)
    eval(cfg.DATASET).get_list(probe_list,os.path.join(cfg.ROOT_PATH,cfg.PROBE),pid_dict)
    num_probe=len(probe_list)
    probe_feats=get_feat(model,probe_list,cfg)
    gallery_feats=get_feat(model,gallery_list,cfg)
    top1=0
    top5=0
    for probe_img in probe_feats:
        distance=[]
        probe_feat=probe_img[3]
        for gallery_img in gallery_feats:
            dist=np.linalg.norm(probe_feat-gallery_img[3])
            if cfg.DATASET == 'Tufts':
                distance.append((dist,gallery_img[1]))
            else:
                distance.append(dist)
        if cfg.DATASET == 'Tufts':
            distance.sort()
            top5_id=[id[1] for id in distance[:5]]
        else:
            p=np.argsort(distance)
            top5_id=p[:5]
        if probe_img[1] in top5_id:
            top5+=1
            if probe_img[1]==top5_id[0]:
                top1+=1
    logger.info(f'val: rank-1: {100*top1/num_probe:.4f}\trank-5: {100*top5/num_probe:.4f}')
    return 100*top1/num_probe



def get_feat(model,img_list,cfg):
    feat_list=[]
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([128,128])])
    input     = torch.zeros(1, 3, 128, 128)
    for img_name in img_list:
        img_path=img_name[0]
        img = Image.open(os.path.join(cfg.ROOT_PATH,img_path)).convert('RGB')
        image = np.array(img)[:, :, ::-1]
        img = Image.fromarray(np.uint8(image))

        img   = transform(img)
        input[0,:,:,:] = img

        input = input.cuda()
        with torch.no_grad():
            input_var   = torch.autograd.Variable(input)
            output = model(input_var)
            features = output[1]
            feat_list.append((img_name[0],img_name[1],img_name[2],features.data.cpu().numpy()[0]))
    return feat_list



if __name__ == '__main__':
    parser=argument_parser()
    args=parser.parse_args()
    logger.info("Command Line Args:", args)
    main(args)