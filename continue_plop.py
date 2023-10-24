import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from model.swinunetr import SwinUNETR
from model.swinunetr_partial_v3 import SwinUNETR as SwinUNETR_partial_v3
from dataset.dataloader_continue import get_loader
from utils import loss
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


torch.multiprocessing.set_sharing_strategy('file_system')

def pod_features_distillation(
    list_attentions_a,
    list_attentions_b,
    normalize=False,
    pod_factor=0.01,
    prepro="pow",
    spp_scales=[1, 2, 4],
    use_pod_schedule=True,
    nb_current_classes=-1,
    nb_new_classes=-1
):
    """A mega-function comprising several features-based distillation.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    device = list_attentions_a[0].device

    assert len(list_attentions_a) == len(list_attentions_b)

    adaptative_pod_factor = 1.0
    normalize_per_scale = False
    pod_factor = 0.0005
    prepro = "pow"
    spp_scales = [1,2,4]
    use_pod_schedule = True
    normalize_per_scale = False

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):

        if i == len(list_attentions_a) - 1: # last attention map
            # assert a.shape[0] == b.shape[0]
            # assert a.shape[2] == b.shape[2]
            # assert a.shape[3] == b.shape[3]
            pass

        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if prepro == "pow": # yes
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif prepro == "none":
            pass
        elif prepro == "abs":
            a = torch.abs(a, 2)
            b = torch.abs(b, 2)
        elif prepro == "relu":
            a = torch.clamp(a, min=0.)
            b = torch.clamp(b, min=0.)

        # if collapse_channels == "local": # yes
        a = _local_pod(
            a, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
        )
        b = _local_pod(
            b, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
        )

        if normalize: # no
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        # difference_function == "frobenius": # yes
        if isinstance(a, list):
            layer_loss = torch.tensor(
                [torch.frobenius_norm(aa - bb, dim=-1) for aa, bb in zip(a, b)]
            ).to(device)
        else:
            layer_loss = torch.frobenius_norm(a - b, dim=-1)

        assert torch.isfinite(layer_loss).all(), layer_loss
        assert (layer_loss >= 0.).all(), layer_loss

        layer_loss = torch.mean(adaptative_pod_factor * layer_loss) # adaptative_pod_factor=1
        if pod_factor <= 0.:
            continue

        layer_loss = pod_factor * layer_loss # pod_factor 0.0005
        if use_pod_schedule: # yes
            layer_loss = layer_loss * math.sqrt(nb_current_classes / nb_new_classes)
        loss += layer_loss

    return loss / len(list_attentions_a)

def _local_pod(x, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale**2

        for i in range(scale):
            for j in range(scale):
                for z in range(scale):
                    tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k, z * k:(z + 1) * k]

                    horizontal_pool = tensor.mean(dim=-1).view(b, -1) # b,c,d,h,w
                    vertical_pool = tensor.mean(dim=-2).view(b, -1)
                    dimension_pool = tensor.mean(dim=-3).view(b, -1)

                    if normalize_per_scale is True:
                        horizontal_pool = horizontal_pool / nb_regions
                        vertical_pool = vertical_pool / nb_regions
                        dimension_pool = dimension_pool / nb_regions

                    if normalize:
                        horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                        vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                    emb.append(horizontal_pool)
                    emb.append(vertical_pool)
                    emb.append(dimension_pool)
    return torch.cat(emb, dim=1)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = loss.BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, organ_list):
        total_loss = []
        predict = F.sigmoid(predict)

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            for organ in organ_list:
                dice_loss = self.dice(predict[b, organ-1], target[b, organ-1])
                total_loss.append(dice_loss)
            
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]


class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, organ_list):
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'
        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            for organ in organ_list:
                ce_loss = self.criterion(predict[b, organ-1], target[b, organ-1])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]


def train(args, train_loader, model, old_model, optimizer, loss_func_dice, loss_func_bce, loss_func_ce):
    model.train()
    loss_bce_ave = 0
    loss_ce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        new_outputs = model(x)
        with torch.no_grad():
            old_outputs = old_model(x, return_feature=True)
        logit_map = new_outputs[-1]

        dist_loss = pod_features_distillation(
            new_outputs[:-2],
            old_outputs[:-2],
            nb_current_classes=args.cur_classes,
            nb_new_classes=args.new_classes
        )

        if args.out_nonlinear == 'sigmoid':
            term_seg_Dice = loss_func_dice.forward(logit_map, y, args.organ_list)
            term_seg_BCE = loss_func_bce.forward(logit_map, y, args.organ_list)
            loss = term_seg_BCE + term_seg_Dice + dist_loss
            loss_bce_ave += term_seg_BCE.item()
            loss_dice_ave += term_seg_Dice.item()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                    args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
            )
        elif args.out_nonlinear == 'softmax':
            b, c, d, h, w = y.shape
            label = y.new_zeros((b, d, h, w), dtype=torch.long)
            for icls in range(c):
                label[y[:, icls] == 1] = icls + 1
            term_seg_Dice = loss_func_dice.forward(logit_map[:, 1:], y, args.organ_list)
            term_seg_CE = loss_func_ce(logit_map, label)
            loss = term_seg_Dice + term_seg_CE + dist_loss
            loss_ce_ave += term_seg_CE.item()
            loss_dice_ave += term_seg_Dice.item()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                    args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_CE.item())
            )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    
    return loss_dice_ave / len(epoch_iterator), loss_bce_ave / len(epoch_iterator), loss_ce_ave / len(epoch_iterator)


def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model
    if args.model == 'swinunetr_partial':
        model = SwinUNETR_partial_v3(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            encoding=args.trans_encoding,
        )
        old_model = SwinUNETR_partial_v3(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            encoding=args.trans_encoding,
        )
    elif args.model == 'swinunetr':
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
        )
        old_model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
        )
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    pretrain_checkpoint = torch.load(args.pretrain)
    if 'state_dict' in pretrain_checkpoint:
        model_dict = pretrain_checkpoint["state_dict"]
    else:
        model_dict = pretrain_checkpoint['net']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        model_dict, "module."
    )
    
    for key in model_dict.keys():
        # if 'out' not in key:
        #     store_dict[key] = model_dict[key]
        # else:
        #     print(f'{key} is not in model state dict')
        store_dict[key] = model_dict[key]

    model.load_state_dict(store_dict)
    print('Use pretrained weights')

    if args.model == 'swinunetr_partial' and args.trans_encoding == 'word_embedding':
        word_embedding = torch.load(args.word_embedding)
        model.organ_embedding.data = word_embedding.float()
        print('load word embedding')

    model.to(args.device)
    model.train()
    old_model.to(args.device)
    old_model.eval()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device])
        old_model = DistributedDataParallel(old_model, device_ids=[args.device])

    # criterion and optimizer
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_func_dice = DiceLoss().to(args.device)
    loss_func_bce = Multi_BCELoss().to(args.device)
    loss_func_ce = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # if args.model == 'swinunetr_partial':
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # elif args.model == 'swinunetr':
    #     model_params = [v for k, v in model.named_parameters() if 'out' not in k]
    #     class_params = model.module.out.parameters()
    #     optimizer = torch.optim.AdamW(
    #         [{'params': class_params, 'lr': 1e-2},
    #         {'params': model_params}],
    #         lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    # load old model
    if args.old_model:
        checkpoint = torch.load(args.old_model)
        if args.dist:
            old_model.load_state_dict(checkpoint['net'])
        else: 
            model_dict = checkpoint['net']
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                model_dict, "module."
            )
            old_model.load_state_dict(model_dict)
        for name, parameter in old_model.named_parameters():
            parameter.requires_grad = False

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            model_dict = checkpoint['net']
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                model_dict, "module."
            )
            model.load_state_dict(model_dict)
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.log_name))
        print('Writing Tensorboard logs to ', os.path.join(args.log_dir, args.log_name))
    
    if not os.path.isdir(os.path.join(args.log_dir, args.log_name)):
        os.mkdir(os.path.join(args.log_dir, args.log_name))

    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce, loss_ce = train(args, train_loader, model, old_model, optimizer, loss_func_dice, loss_func_bce, loss_func_ce)
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('train_ce_loss', loss_ce, args.epoch)
            # writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if (args.epoch % args.store_num == 0 and args.epoch != 0) and rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            torch.save(checkpoint, os.path.join(args.log_dir, args.log_name, f'epoch_{args.epoch}.pth'))
            print('save model success')

        args.epoch += 1

    if args.dist:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_dir', default='output', help='Log directory.')
    parser.add_argument('--log_name', type=str, required=True, help='Experiment name under the log dir.')
    ## model load
    parser.add_argument('--model', type=str, choices=['swinunetr', 'swinunetr_partial'])
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--old_model', default=None, help='The path of checkpoint of old model')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth', 
                        help='The path of word embedding')
    parser.add_argument('--out_nonlinear', type=str, choices=['softmax', 'sigmoid'])
    parser.add_argument('--out_channels', type=int)
    ## hyperparameter
    parser.add_argument('--max_epoch', default=2000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=100, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    ### for cross_validation 'cross_validation/PAOT_0' 1 2 3 4
    parser.add_argument('--data_root_path', default='', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--train_data_txt_path', type=str, help='train data txt path.')
    parser.add_argument('--val_data_txt_path', type=str, help='val data txt path.')
    parser.add_argument('--test_data_txt_path', type=str, help='test data txt path.')
    parser.add_argument('--continue_data_txt_path', type=str, help='continue data txt path.')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='conitnue', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                            '07', '08', '09', '12', '13', '10_03', 
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')

    parser.add_argument('--organ_list', nargs='+', type=int, required=True, help='Target training organ ids.')

    ## args for pod
    parser.add_argument('--cur_classes', type=int, help='Number of current classes including new and old.')
    parser.add_argument('--new_classes', type=int, help='Number of new classes.')
    
    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
