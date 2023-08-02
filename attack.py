import sys 
sys.path.append('.')

import torch
import os
import time
from PIL import Image
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from model.lightcnn_v4_MCR import LightCNN_V4
from utils.init import get_config,setup,argument_parser
from utils.model_util import logging,AverageMeter,accuracy,adjust_learning_rate,save_checkpoint,get_confidence
from utils.file_io import PathManager
from utils.datasets.NIR_VISv2 import NIR_VISv2
from utils.datasets.LAMP_HQ import LAMP_HQ
from utils.datasets.Tufts import Tufts
from utils.FAT import earlystop

logger=logging.getLogger("Train")

def adjust_tau(epoch, dynamictau):
    if dynamictau:
        if epoch <= 25:
            tau = 0
        elif epoch <= 45:
            tau = 1
        else:
            tau = 2
    return tau

def get_model(cfg):

    datas_train=eval(cfg.DATASET)(root=cfg.ROOT_PATH,protocols=cfg.PROTOCOLS)
    
    model=LightCNN_V4(num_classes=datas_train.num_classes,classifier=True)
    model = model.cuda()

    # load pretrained lightcnn
    logger.info("loading pretrained lightcnn model '{}'".format(cfg.WEIGHTS))
    checkpoint = torch.load(cfg.WEIGHTS)
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model,datas_train


def classifier_train(cfg,train_loader,model,criterion, optimizer, epoch):
    losses     = AverageMeter()
    dom_top1   = AverageMeter()
    tar_top1 = AverageMeter()


    for i, (input, target, domain) in enumerate(train_loader):

        input      = input.cuda()
        target     = target.cuda()
        domain     = domain.cuda()
        input_var  = torch.autograd.Variable(input)
        domain_var = torch.autograd.Variable(domain)
        
        model.train()
        output = model(input_var)
 
        dom = output[1]
        tar = output[0]
        
        loss   = criterion(dom, domain_var)
        domain_prec1= accuracy(dom.data, domain , topk=(1,))
        target_prec1 = accuracy(tar.data, target , topk=(1,))
        losses.update(loss.item(), input.size(0))
        dom_top1.update(domain_prec1[0].item(), input.size(0))
        tar_top1.update(target_prec1[0].item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % cfg.PRINT_FREQ == 0:
            logger.info(
                '[classifier]Epoch: [{0}][{1:3}/{2}]\t'
                'Loss {loss.avg:>8.4f}\t'
                'dom_Prec@1 {dom_top1.avg:>6.3f}\t'
                .format(
                   epoch, i, len(train_loader),
                    loss=losses, dom_top1=dom_top1,tar_top1=tar_top1))

def attack_train(cfg,train_loader,model,criterion, optimizer, epoch):
    losses     = AverageMeter()
    dom_top1       = AverageMeter()
    tar_top1 = AverageMeter()


    for i, (input, target, domain) in enumerate(train_loader):

        input      = input.cuda()
        target     = target.cuda()
        domain     = domain.cuda()
        input_var  = torch.autograd.Variable(input)
        domain_var = torch.autograd.Variable(domain)
        target_var = torch.autograd.Variable(target)

        output_adv, output_domain, output_target, output_natural, output_index = earlystop(model, input_var, domain_var,target_var, step_size=cfg.STEP_SIZE,
                                                                     epsilon=cfg.EPSILON, perturb_steps=cfg.NUM_STEPS, tau=adjust_tau(epoch, cfg.DYNAMICTAU),
                                                                     randominit_type="uniform_randominit", loss_fn='cent', rand_init=cfg.RAND_INIT, omega=cfg.OMEGA,beta=cfg.BETA)        
        model.train()
        output = model(output_adv)

        dom = output[1]
        tar = output[0]
        
        loss   = criterion(dom, output_domain)
        domain_prec1= accuracy(dom.data, output_domain , topk=(1,))
        target_prec1 = accuracy(tar.data, output_target , topk=(1,))
        losses.update(loss.item(), output_adv.size(0))
        dom_top1.update(domain_prec1[0].item(), output_adv.size(0))
        tar_top1.update(target_prec1[0].item(), output_adv.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % cfg.PRINT_FREQ == 0:
            logger.info(
                '[attack]Epoch: [{0}][{1:3}/{2}]\t'
                'Loss {loss.avg:>8.4f}\t'
                'dom_Prec@1 {dom_top1.avg:>6.3f}\t'
                'tar_Prec@1 {tar_top1.avg:>6.3f}\t'
                .format(
                   epoch, i, len(train_loader),
                    loss=losses, dom_top1=dom_top1,tar_top1=tar_top1))
    
    if epoch == cfg.EPOCHS:
        datas_attack=eval(cfg.DATASET)(root=cfg.ROOT_PATH,protocols=['attack_list'],needimgpath=True)
        attack_loader = data.DataLoader(
            datas_attack,batch_size=cfg.BATH_SIZE, shuffle=True,
            num_workers=cfg.WORKERS, pin_memory=True)
        for i, (input, target, domain, imgPath) in enumerate(attack_loader):

            input      = input.cuda()
            target     = target.cuda()
            domain     = domain.cuda()
            input_var  = torch.autograd.Variable(input)
            domain_var = torch.autograd.Variable(domain)
            target_var = torch.autograd.Variable(target)

            output_adv, output_domain, output_target, output_natural, output_index = earlystop(model, input_var, domain_var,target_var, step_size=cfg.STEP_SIZE,
                                                                         epsilon=cfg.EPSILON, perturb_steps=cfg.NUM_STEPS, tau=adjust_tau(epoch, cfg.DYNAMICTAU),
                                                                         randominit_type="uniform_randominit", loss_fn='cent', rand_init=cfg.RAND_INIT, omega=cfg.OMEGA,beta=cfg.BETA)
            toPIL = transforms.ToPILImage()
            root_path = cfg.ATTACK_PATH       
            imgpath = np.array(imgPath)
            index = output_index.data.cpu().numpy()
            imgpath = imgpath[index]
            imgpath = list(imgpath)
            for k in range(len(input)):
                dir_path = os.sep.join(imgpath[k].split(os.sep)[:-1])
                if not PathManager.exists(os.path.join(root_path,dir_path)):
                        PathManager.mkdirs(os.path.join(root_path,dir_path))
                img = toPIL(output_adv[k])
                image = np.array(img)[:, :, ::-1]
                img = Image.fromarray(np.uint8(image))
                img.save(os.path.join(root_path,imgpath[k]))


def val_model(model,cfg):
    model.eval()
    image_list=[]
    top1=0
    pid_dict = {}
    for test in cfg.TEST:
        eval(cfg.DATASET).get_list(image_list,os.path.join(cfg.ROOT_PATH,'Protocol',test+'.txt'),pid_dict)
    random.shuffle(image_list)
    img_num=len(image_list)
    for img_name in image_list:
        img_path=img_name[0]
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([128,128])])
        input     = torch.zeros(1, 3, 128, 128)
        img = Image.open(os.path.join(cfg.ROOT_PATH,img_path)).convert('RGB')
        image = np.array(img)[:, :, ::-1]
        img = Image.fromarray(np.uint8(image))
        img   = transform(img)
        input[0,:,:,:] = img
        input = input.cuda()
        with torch.no_grad():
            input_var   = torch.autograd.Variable(input)
            output = model(input_var)
            domain = output[1]
            if img_name[2]==np.argmax(domain.data.cpu().numpy()[0]):
                top1+=1
    logger.info(f'domain_val: rank-1: {100*top1/img_num:.4f}')
    return 100*top1/img_num

def updata_list(model,cfg):
    attackpath = cfg.ROOT_PATH + '/Protocol/attack_list.txt'
    model.eval()
    unlabel_list=[]
    pid_dict = {}
    for unlabel in cfg.UNLABEL:
        eval(cfg.DATASET).get_list(unlabel_list,os.path.join(cfg.ROOT_PATH,'Protocol',unlabel+'.txt'),pid_dict)
    pid_list=list(pid_dict.keys())
    confidence_list=get_confidence(model,unlabel_list,cfg)
    with open(attackpath,'w') as f:
        for line in confidence_list:
            if cfg.DATASET == 'LAMP_HQ':
                f.writelines(line[0]+' '+ pid_list[line[3][0]]  + '\n')
            elif cfg.DATASET == 'NIR_VISv2':
                f.writelines(line[0].replace(os.sep,'\\')+' '+ pid_list[line[3][0]]+'\n')
            else:
                f.writelines(line[0]+' '+ pid_list[line[3][0]]+'\n')

def train(model,datas_train,cfg): 
    train_loader = data.DataLoader(
        datas_train,batch_size=cfg.BATH_SIZE, shuffle=True,
        num_workers=cfg.WORKERS, pin_memory=True)
    criterion=nn.CrossEntropyLoss().cuda()
    params= []
    for name, value in model.named_parameters():
        if 'classifiar' in name :
            params+= [{'params': value, 'lr': cfg.CLASS_LR}]
        
    domain_optimizer=torch.optim.SGD(params, cfg.CLASS_LR,momentum=cfg.MOENTUM,weight_decay=cfg.WEIGHT_DECAY)
    optimizer=torch.optim.Adam(params, cfg.LR,weight_decay=cfg.WEIGHT_DECAY)
    max_acc = 0
    logger.info('---------------start classifier train---------------')
    for epoch in range(cfg.CLASS_EPOCH):
        adjust_learning_rate(cfg.CLASS_LR, domain_optimizer, epoch,step = 3)
        classifier_train(cfg,train_loader, model, criterion, domain_optimizer, epoch+1)
        if epoch%cfg.VAL_FREQ==0:
            acc = val_model(model,cfg)
            if acc > max_acc == 0:
                save_name = cfg.OUTPUT_PATH  + "/classifier_" +str(acc)[:6] + str(epoch+1)+'_'+cfg.DATASET + '.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': 'classifier',
                    'state_dict': model.state_dict(),
                    'prec1': acc,
                    'time' : time.time()
                }, save_name)
                max_acc = acc
                best = model.state_dict()
    logger.info('---------------start attack train---------------')
    model.load_state_dict(best)
    prec1 = val_model(model,cfg)
    params= []
    for name, value in model.named_parameters():
        if 'classifiar' in name :
            params+= [{'params': value, 'lr': cfg.LR}]
    updata_list(model,cfg)
    datas_train=eval(cfg.DATASET)(root=cfg.ROOT_PATH,protocols=['attack_list'])
    train_loader = data.DataLoader(datas_train,batch_size=cfg.BATH_SIZE, shuffle=True,num_workers=cfg.WORKERS, pin_memory=True)
    for epoch in range(cfg.EPOCHS):
        attack_train(cfg,train_loader, model, criterion, optimizer, epoch+1)
        if epoch%cfg.VAL_FREQ==0:
            prec1 = val_model(model,cfg)
            if epoch % 10 == 0:
                save_name = cfg.OUTPUT_PATH  + "/attack_" +str(prec1)[:6] + str(epoch+1)+'_'+cfg.DATASET + '.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': 'attack',
                    'state_dict': model.state_dict(),
                    'prec1': prec1,
                    'time' : time.time()
                }, save_name)
    return model.state_dict()
    
                
if __name__ == '__main__':
    cudnn.benchmark=True
    parser=argument_parser()
    args=parser.parse_args()
    cfg_path = args.config_file
    cfg=get_config(cfg_path)
    
    setup(cfg)
    model,datas_train=get_model(cfg)
    train(model,datas_train,cfg)
