import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import os
import random


def default_loader(path):
    img = Image.open(path).convert('RGB')
    image = np.array(img)[:, :, ::-1]
    img = Image.fromarray(np.uint8(image))
    return img


def default_list_reader(label_dict,root,protocols):
    imgList = []
    for protocol in protocols:
        protocol_path=os.path.join(root,'Protocol',protocol+'.txt')
        NIR_VISv2.get_list(imgList,protocol_path,label_dict)

    random.seed(1)
    random.shuffle(imgList)
    return imgList

class NIR_VISv2(data.Dataset):
    def __init__(self, root, protocols ,list_reader=default_list_reader, loader=default_loader,pid_dict={},needimgpath=False):
        self.root= root
        self.label_dict=pid_dict
        self.imgList   = list_reader(self.label_dict,root,protocols)
        self.num_classes = len(self.label_dict)
        self.loader    = loader
        self.needimgpath = needimgpath
        self.size = 128
        self.transform = transforms.Compose([
            transforms.Resize([self.size,self.size]),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, index):
        imgPath, pid, domain = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        img = self.transform(img)
        if self.needimgpath:
            return img,int(pid),domain,imgPath
        else:
            return img,int(pid),domain

    def __len__(self):
        return len(self.imgList)

    def get_list(imgList,protocol,label_dict):
        with open(protocol, 'r') as file:
            for line in file.readlines():
                line = line.strip('\n')
                if 'NIR' in line:
                    domain=0
                else:
                    domain=1
                if len(line.split(' ')) == 1:
                    imgPath=os.path.join(line.replace('\\',os.sep))
                    pid=line.split('\\')[-2]
                else:
                    imgPath=os.path.join(line.split(' ')[0].replace('\\',os.sep))
                    pid=line.split(' ')[1]
                if not label_dict:
                    label_dict[pid]=0
                if pid in label_dict.keys():
                    label = label_dict[pid]
                else:
                    label_dict[pid]=max(label_dict.values())+1
                    label = label_dict[pid]
                imgList.append((imgPath, label, domain))

if __name__=='__main__':
    pro=NIR_VISv2('datasets/NIR-VIS-2.0')
    print(pro[1])