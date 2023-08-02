from turtle import color
import torch
import time
import logging
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.datasets.LAMP_HQ import LAMP_HQ
from utils.datasets.NIR_VISv2 import NIR_VISv2
from utils.datasets.Tufts import Tufts
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import os


logger=logging.getLogger('Train')

def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def save_checkpoint(state,filename):
    torch.save(state, filename)


def validate(test_loader, model, criterion):
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()
    middle1_losses = AverageMeter()
    middle2_losses = AverageMeter()
    middle3_losses = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, pid, domain) in enumerate(test_loader):
        input = input.cuda()
        pid = pid.cuda()
        with torch.no_grad():
            input_var  = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(pid)

            # compute output
            output, fc ,(middle_fc1,middle_fc2,middle_fc3) ,(middle_out1,middle_out2,middle_out3), (middle_fea1,middle_fea2,middle_fea3,final_fea)= model(input_var)

            loss   = criterion(output, target_var)
            middle1_loss = criterion(middle_out1, target_var)
            middle2_loss = criterion(middle_out2, target_var)
            middle3_loss = criterion(middle_out3, target_var)


            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, pid, topk=(1,5))
            middle1_prec1 = accuracy(middle_out1.data, pid, topk=(1,))
            middle2_prec1 = accuracy(middle_out2.data, pid, topk=(1,))
            middle3_prec1 = accuracy(middle_out3.data, pid, topk=(1,))
            losses.update(loss.item(), input.size(0))
            middle1_losses.update(middle1_loss.item(),input.size(0))
            middle2_losses.update(middle2_loss.item(),input.size(0))
            middle3_losses.update(middle3_loss.item(),input.size(0))
            top1.update(prec1.item(), input.size(0))
            middle1_top1.update(middle1_prec1[0], input.size(0))
            middle2_top1.update(middle2_prec1[0], input.size(0))
            middle3_top1.update(middle3_prec1[0], input.size(0))
            top5.update(prec5.item(), input.size(0))


    logger.info('Test set: Average loss: {}, Accuracy: ({})'.format(losses.avg, top1.avg))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def adjust_learning_rate(lr,optimizer, epoch ,step  = 20):
    scale = 0.457305051927326
    lr = lr * (scale ** (epoch // step))
    if (epoch != 0) and (epoch % step == 0) or epoch == 5:
        logger.info(f'Change lr to: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale
    return lr

def adjust_dropout_rate(cfg,model,epoch):
    dropout_rates = cfg.DROPOUT_RATE
    step = 4
    rate = [dropout_rates[i] ** (0.8 ** (epoch // step)) if dropout_rates[i] ** (0.8 ** (epoch // step))<=cfg.MAX_DROPOUT[i] else cfg.MAX_DROPOUT[i] for i in range(len(dropout_rates))]
    if (epoch != 0) and (epoch % step == 0):
        logger.info(f'Change dropout to {rate}')
        model.dropout_rate = rate
    return rate[3]
     
def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def base_train(cfg,train_loader, model, criterion, optimizer, epoch):
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()
    total_losses = AverageMeter()
    model.train()

    for i, (input, target, domain) in enumerate(train_loader):

        input      = input.cuda()
        target     = target.cuda()
        input_var  = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, fc ,(middle_fea1,middle_fea2,middle_fea3) ,(middle_out1,middle_out2,middle_out3)= model(input_var)
        loss   = criterion(output, target_var)
        middle1_loss = criterion(middle_out1, target_var)
        middle2_loss = criterion(middle_out2, target_var)
        middle3_loss = criterion(middle_out3, target_var)

        kl_loss1 = F.kl_div(F.log_softmax(middle_fea1 / cfg.T, dim=1), F.softmax(fc / cfg.T, dim=1), reduction='sum') * (cfg.T**2) / fc.shape[0]
        kl_loss2 = F.kl_div(F.log_softmax(middle_fea2 / cfg.T, dim=1), F.softmax(fc / cfg.T, dim=1), reduction='sum') * (cfg.T**2) / fc.shape[0]
        kl_loss3 = F.kl_div(F.log_softmax(middle_fea3 / cfg.T, dim=1), F.softmax(fc / cfg.T, dim=1), reduction='sum') * (cfg.T**2) / fc.shape[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        total_loss =  cfg.ALPHA * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                    (1 - cfg.ALPHA) * (kl_loss1 + kl_loss2 + kl_loss3)
        total_losses.update(total_loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        if i % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1:3}/{2:3}]\t'
                'Loss {loss.avg:>8.4f}\t'
                'total_loss {total_loss.avg:>8.4f}\t'
                'Prec@1 {top1.avg:>6.3f}\t'
                'Prec@5 {top5.avg:>6.3f}\t'
                .format(
                   epoch, i, len(train_loader),
                    loss=losses, top1=top1, top5=top5,total_loss=total_losses))

def cross_entropy_loss(class_outputs, gt_classes, eps=0.1, alpha=0.2):
    num_classes = class_outputs.size(1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt

    return loss


def update_imglist(model,cfg,pid_dict):
    imglistpath = cfg.UNLABEL_PROTOCOL
    attackpath = cfg.ATTACK_PATH
    model.eval()
    unlabel=[]
    for i in cfg.UNLABEL:
        eval(cfg.DATASET).get_list(unlabel,os.path.join(cfg.ROOT_PATH,'Protocol',i+'.txt'),pid_dict)
    pid_list=list(pid_dict.keys())
    confidence_list=get_confidence(model,unlabel,cfg)
    with open(imglistpath,'w') as f:
        for line in confidence_list:
            if line[3][1] > cfg.TAU:
                if cfg.DATASET == 'LAMP_HQ':
                    f.writelines(line[0]+' '+ pid_list[line[3][0]]  + '\n')
                elif cfg.DATASET == 'NIR_VISv2':
                    f.writelines(line[0].replace(os.sep,'\\')+' '+ pid_list[line[3][0]]+'\n')
                else:
                    f.writelines(line[0]+' '+ pid_list[line[3][0]]+'\n')
    with open(attackpath,'w') as f:
        for line in confidence_list:
            if line[3][1] > cfg.TAU:
                if cfg.DATASET == 'LAMP_HQ':
                    f.writelines(cfg.ATTACK_HEAD + '/'+ line[0]+' '+ pid_list[line[3][0]]  + '\n')
                elif cfg.DATASET == 'NIR_VISv2':
                    f.writelines(cfg.ATTACK_HEAD+'\\' +  line[0].replace(os.sep,'\\')+' '+ pid_list[line[3][0]]+'\n')
                else:
                    f.writelines(cfg.ATTACK_HEAD+'/'+line[0]+' '+ pid_list[line[3][0]]+'\n')

    

def get_confidence(model,img_list,cfg):
    confidence_list=[]
    count     = 0
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([128,128])])
    input     = torch.zeros(1, 3, 128, 128)
    for img_name in img_list:
        img_path=img_name[0]
        count = count + 1
        img = Image.open(os.path.join(cfg.ROOT_PATH,img_path)).convert('RGB')
        image = np.array(img)[:, :, ::-1]
        img = Image.fromarray(np.uint8(image))


        img   = transform(img)
        input[0,:,:,:] = img
        input = input.cuda()
        with torch.no_grad():
            input_var   = torch.autograd.Variable(input)
            output = model(input_var)
            confidence = output[0]
            confidence = F.softmax(confidence,dim=1)
            confidence = confidence.data.cpu().numpy()[0]
            confidence_list.append((img_name[0],img_name[1],img_name[2],(np.argmax(confidence),max(confidence), True if np.argmax(confidence)==img_name[1] else False)))
    return confidence_list