import os
import sys
import logging
import torch
import time

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data

sys.path.append('.')
from model.lightcnn_v4_MCR import LightCNN_V4
from utils.init import argument_parser,get_config,setup
from utils.model_util import load_model,adjust_dropout_rate,adjust_learning_rate,base_train,save_checkpoint,update_imglist
from utils.datasets.NIR_VISv2 import NIR_VISv2
from utils.datasets.LAMP_HQ import LAMP_HQ
from utils.datasets.Tufts import Tufts
from utils.val import val_model




logger=logging.getLogger("Train")
def get_model(cfg):

    cudnn.benchmark=True
    protocols=cfg.PRE_PROTOCOL
    datas_train=eval(cfg.DATASET)(root=cfg.ROOT_PATH,protocols=protocols)
    model=LightCNN_V4(num_classes=datas_train.num_classes,dropout_rate=cfg.DROPOUT_RATE)

    if cfg.CUDA:
        model = model.cuda()

    # load pretrained lightcnn
    logger.info("loading pretrained lightcnn model '{}'".format(cfg.WEIGHTS))
    model = load_model(model,cfg.WEIGHTS)

    return model,datas_train



def train(model,datas_train,cfg,val_cfg): 
    
    pid_dict=datas_train.label_dict

    criterion=nn.CrossEntropyLoss()
    if cfg.CUDA:
        criterion.cuda()

    optimizer=torch.optim.SGD(model.parameters(), cfg.LR,
                            momentum=cfg.MOENTUM,
                            weight_decay=cfg.WEIGHT_DECAY)

    if cfg.PERTRAIN:
        pre_datas_train=eval(cfg.DATASET)(root=cfg.ROOT_PATH,protocols=cfg.PRE_PROTOCOL)
        pre_train_loader = data.DataLoader(pre_datas_train,batch_size=cfg.BATH_SIZE, shuffle=True,
            num_workers=cfg.WORKERS, pin_memory=True)
        params_pretrain = []
        for name, value in model.named_parameters():
            if 'fc2_' in name :
                params_pretrain += [{'params': value, 'lr': cfg.PRE_LR}]
            elif 'fc' in name :
                params_pretrain += [{'params': value, 'lr': 0.1*cfg.PRE_LR}]
            elif 'middle' in name:
                params_pretrain += [{'params': value, 'lr': cfg.PRE_LR}]

        optimizer_pretrain=torch.optim.SGD(params_pretrain, cfg.PRE_LR,
                                momentum=cfg.MOENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)

        logger.info('------------start fc2 pertrain------------')
        for epoch in range(cfg.FC2_EPOCHS):
            adjust_learning_rate(cfg.PRE_LR, optimizer_pretrain, epoch)
            base_train(cfg,pre_train_loader, model, criterion, optimizer_pretrain, epoch+1)

    max_acc=val_model(model,val_cfg)
    logger.info('-------------start train-------------')
    for epoch in range(0,cfg.EPOCHS):

        update_imglist(model,cfg,pid_dict)
        protocols=cfg.PROTOCOLS
        datas_train=eval(cfg.DATASET)(root=cfg.ROOT_PATH,protocols=protocols,pid_dict=pid_dict)
        train_loader = data.DataLoader(
            datas_train,batch_size=cfg.BATH_SIZE, shuffle=True,
            num_workers=cfg.WORKERS, pin_memory=True)

        adjust_learning_rate(cfg.LR, optimizer, epoch, cfg.ADJUST_EPOCH)
        adjust_dropout_rate(cfg,model,epoch)

        # train for one epoch
        base_train(cfg, train_loader, model, criterion, optimizer, epoch+1)

        if epoch%cfg.VAL_FREQ==0:
            prec1 = val_model(model,val_cfg)
            if max_acc <= prec1:
                max_acc = prec1
                save_name = cfg.OUTPUT_PATH +str(prec1)[:6] +cfg.DATASET + '.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg.MODEL,
                    'state_dict': model.state_dict(),
                    'prec1': prec1,
                    'time' : time.time()
                }, save_name)
    return max_acc




if __name__=="__main__":
    parser=argument_parser()
    args=parser.parse_args()
    cfg_path = args.config_file
    cfg=get_config(cfg_path)
    val_cfg = get_config(cfg.VAL_PATH)
    setup(cfg)
    model,datas_train=get_model(cfg)
    train(model,datas_train,cfg,val_cfg)