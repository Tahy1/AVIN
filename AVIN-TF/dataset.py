import json
import os
import os.path as osp
import pickle as pk
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import librosa

class AVParis(Dataset):
    def __init__(self, audio_root, vision_root, json_file, mode, num='full'):
        self.mode = mode
        self.audio_root = audio_root
        self.vision_root = vision_root
        with open(json_file, 'r') as fr:
            self.dataset = json.load(fr)
    
        self.anchors = []
        if num == 'full':
            for uuid in self.dataset:
                vision_segs = osp.join(self.vision_root, uuid)
                instances = sorted(os.listdir(vision_segs))
                self.anchors.extend([[uuid, i] for i in instances])
        else:
            for uuid in self.dataset:
                vision_segs = osp.join(self.vision_root, uuid)
                instances = sorted(os.listdir(vision_segs))[:num]
                self.anchors.extend([[uuid, i] for i in instances])

        if self.mode == 'train':
            img_transform_list = [transforms.Resize((256, 256)),
                                  transforms.RandomCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                       std=(0.229, 0.224, 0.225))]
        elif self.mode in ['val', 'test']:
            img_transform_list = [transforms.Resize((224, 224)),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                       std=(0.229, 0.224, 0.225))]
        self.img_transform = transforms.Compose(img_transform_list)
    
    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        uuid_anchor, instance_anchor = self.anchors[index]
        audio_anchor_path = osp.join(
            self.audio_root, uuid_anchor, instance_anchor+'.wav')
        (audio_anchor, _) = librosa.core.load(audio_anchor_path, sr=32000, mono=True)
        vision_anchor_fold = osp.join(
            self.vision_root, uuid_anchor, instance_anchor)
        vision_anchor_path = osp.join(
            vision_anchor_fold, random.choice(os.listdir(vision_anchor_fold)))
        vision_anchor = Image.open(vision_anchor_path)
        vision_anchor = self.img_transform(vision_anchor)
        ret = [audio_anchor, vision_anchor]

        return ret

class Testset(Dataset):
    def __init__(self, audio_root, vision_root, json_file, retmap=True, retpath=False):
        self.audio_root = audio_root
        self.vision_root = vision_root
        self.retmap = retmap
        self.retpath = retpath
        with open(json_file, encoding='utf-8') as fr:
            self.data_dict = json.load(fr)
        self.data_list = sorted(self.data_dict.keys())
        img_transform_list = [transforms.Resize((224, 224)), 
                              transforms.ToTensor(),
                              transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                   std=(0.229, 0.224, 0.225))]
        self.img_transform = transforms.Compose(img_transform_list)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        key = self.data_list[index]
        img_path = osp.join(self.vision_root, key+'.jpg')
        img = Image.open(img_path)
        img_data = self.img_transform(img)
        audio_path = osp.join(self.audio_root, key+'.wav')
        (audio_data, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        ret = [audio_data, img_data]
        if self.retmap:
            _, h, w = img_data.shape
            gt = np.zeros([h, w])
            boxes = self.data_dict[key]
            boxes = np.clip(boxes, 0, 1)
            boxes = (boxes * [w, h, w, h]).astype('int')
            # xmin, ymin, xmax, ymax
            for box in boxes:
                submap = np.zeros([h, w])
                xmin, ymin, xmax, ymax = box
                submap[ymin:ymax, xmin:xmax] = 1
                gt += submap
            gt[gt>0] = 1
            ret.append(gt)
        if self.retpath:
            ret.append(img_path)
        return ret
