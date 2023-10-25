from dataset import AVParis, Testset
from torch.utils.data import DataLoader
from models.avmodels import AVIN

import yaml
import os
import argparse
import torch
import os.path as osp
import utils
import time

import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np

def create_optimizer(model):
    params_group = [
        {'params': model.v_encoder.parameters(), 'lr': 1e-5}, 
        {'params': model.v_proj.parameters(), 'lr': 1e-4},
        {'params': model.a_proj.parameters(), 'lr': 1e-4}
    ]
    return torch.optim.Adam(params_group, betas=(0.9, 0.999), weight_decay=1e-4)

def arguments():
    parser = argparse.ArgumentParser(description='AVIN-RN Training')
    parser.add_argument('cfg', help='configuration')
    parser.add_argument('--w', type=str, default=None)
    parser.add_argument('--tp', type=float, default=0.3)
    parser.add_argument('--tn', type=float, default=0.5)
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.cfg))
    if args.w == None:
        args.w = cfg['work_dir']
    args.num_epochs = cfg['num_epochs']
    args.batch_size = cfg['batch_size']
    args.seed = cfg['seed']
    args.train_index = cfg['train_index']
    args.audio_root_train = cfg['audio_root_train']
    args.vision_root_train = cfg['vision_root_train']
    args.audio_root_val = cfg['audio_root_val']
    args.vision_root_val = cfg['vision_root_val']
    args.box_json = cfg['box_json']
    args.workers = cfg['workers']
    args.logging_iters = cfg['logging_iters']
    args.world_size = torch.cuda.device_count()
    return args

def ddp_setup(rank, world_size):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1347'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def main_worker(gpu, args):
    best_ciou = 0.
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ddp_setup(gpu, args.world_size)

    if gpu == 0:
        # prepare logger
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(args.w, f'{timestamp}.log')
        logger = utils.get_root_logger(log_file, 'AVIN-RN')
        args_dict = vars(args)
        for key, value in args_dict.items():
            logger.info('%s: %s'%(key, value))

    # prepare averager
    avg = utils.Averager()
    
    train_dset = AVParis(args.audio_root_train, args.vision_root_train, args.train_index, mode='train', num=1)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset, seed=args.seed) # drop_last = ?
    train_loader = DataLoader(dataset=train_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True) # drop_last = ?
    val_solo_dset = Testset(args.audio_root_val, args.vision_root_val, args.box_json)
    val_solo_loader = DataLoader(dataset=val_solo_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = AVIN(args.tp, args.tn)
    optimizer = create_optimizer(model)
    for name, module in model._modules.items():
        if next(module.parameters(), None) is None:
            module = module.cuda(gpu)
        elif all(not p.requires_grad for p in module.parameters()):
            module = module.cuda(gpu)
        else:
            module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module).cuda(gpu)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[gpu])
        model._modules[name] = module

    for epoch in range(1, args.num_epochs+1):
        train_sampler.set_epoch(epoch-1)

        if gpu == 0:
            logger.info('start train epoch %2d:'%epoch)
        avg.reset_all()
        model.train()
        for iter, (audio, vision) in enumerate(train_loader, 1):
            bs = audio.shape[0]
            audio = audio.cuda(non_blocking=True)
            vision = vision.cuda(non_blocking=True)

            optimizer.zero_grad()
            loss1, loss2 = model(vision, audio, mode='train')
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            avg.update(float(loss), 0, bs)

            # logging iter
            if iter % args.logging_iters == 0 and gpu == 0:
                loss, _ = avg.average('iter')
                logger.info('ITER epoch:%3d, iter:%3d/%d, loss:%.4f'%(epoch, iter, len(train_loader), loss))
                avg.reset('iter')

        # logging epoch
        loss, _ = avg.average('epoch')
        if gpu == 0:
            logger.info('EPOCH loss:%.4f'%(loss))

        # evaluating location net
        if gpu == 0:
            logger.info('evaluating')
        model.eval()
        avmaps = []
        gtmaps = []
        with torch.no_grad():
            for iter, data in enumerate(val_solo_loader, 1):
                audio_data, image_data, gtmap = data
                audio_data = audio_data.cuda(non_blocking=True)
                image_data = image_data.cuda(non_blocking=True)
                avmap = model(image_data, audio_data, mode='test')
                avmaps.append(avmap)
                gtmaps.append(gtmap)
        gtmaps = torch.squeeze(torch.cat(gtmaps, 0)).cpu()
        avmaps = torch.squeeze(torch.cat(avmaps, 0)).cpu()
        thres = 0.5
        if gpu == 0:
            ciou, auc, nsa = utils.metric(avmaps, gtmaps, thres)
            logger.info('epoch:%3d, eval ciou:%.4f, eval auc:%.4f, eval nsa:%.4f'%(epoch, ciou, auc, nsa))
            save_folder = osp.join(args.w, 'ckpt')
            save_path = osp.join(save_folder, '%.3d.pth'%epoch)
            torch.save(model.state_dict(), save_path)
            if ciou > best_ciou:
                best_ciou = ciou
                logger.info('best checkpoint saved sucessfully')
                save_path = osp.join(save_folder, 'best.pth')
                torch.save(model.state_dict(), save_path)
    dist.destroy_process_group()

if __name__ == '__main__':
    args = arguments()
    
    # prepare work dir
    utils.mkdir_or_exist(args.w)
    save_folder = osp.join(args.w, 'ckpt')
    utils.mkdir_or_exist(save_folder)

    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
