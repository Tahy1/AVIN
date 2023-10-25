from dataset import Testset
from torch.utils.data import DataLoader
from models.avmodels import AVIN

import yaml
import argparse
import torch

import time

parser = argparse.ArgumentParser(description='AVIM-RN Training')
parser.add_argument('cfg', help='configuration')
parser.add_argument('--w', type=str, default='none')
parser.add_argument('--tp', type=float, default=0.3)
parser.add_argument('--tn', type=float, default=0.5)
parser.add_argument('--train_file', type=str, default='none')
parser.add_argument('--audio_root_val', type=str, default='none')
parser.add_argument('--vision_root_val', type=str, default='none')
parser.add_argument('--box_json', type=str, default='none')
parser.add_argument('--data_type', type=str, default='none')
parser.add_argument('--num_epochs', type=int, default='100')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))
audio_root_val = cfg['audio_root_val']
vision_root_val = cfg['vision_root_val']
box_json = cfg['box_json']
bz = cfg['bz']
workers = cfg['workers']
work_dir =  cfg['work_dir']
data_type = cfg['data_type']
num_epochs = args.num_epochs
if args.audio_root_val !='none':
    audio_root_val = args.audio_root_val
if args.vision_root_val !='none':
    vision_root_val = args.vision_root_val
if args.box_json != 'none':
    box_json = args.box_json
if args.data_type != 'none':
    data_type = args.data_type
tp = args.tp
tn = args.tn
tau = 0.03
out_dim = 512
dropout_img = 0.9
dropout_aud = 0
def main():
    val_solo_dset = Testset(audio_root_val, vision_root_val, box_json, data_type)
    val_solo_loader = DataLoader(dataset=val_solo_dset, batch_size=bz, shuffle=False, num_workers=workers, pin_memory=True)
    
    model = AVIN(0.0,0.0).cuda()
    model.eval()
    cnt = 0
    with torch.no_grad():
        for iter, data in enumerate(val_solo_loader, 1):
            audio_data, image_data, gtmap = data
            if iter > 1:
                cnt += audio_data.shape[0]
            if iter == 2:
                start = time.time()
            audio_data = audio_data.cuda(non_blocking=True)
            image_data = image_data.cuda(non_blocking=True)
            avmap = model(image_data, audio_data, mode='test')
    end = time.time()
    print(end - start , cnt)
    print(cnt / (end-start))
if __name__ == '__main__':
    main()
