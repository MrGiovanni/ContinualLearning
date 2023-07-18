import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import SimpleITK as sitk

from monai.losses import DiceCELoss
from monai.data import DataLoader, Dataset, list_data_collate, decollate_batch
from monai.transforms import AsDiscrete, Compose, Invertd, SaveImaged
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
# from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)

from model.swinunetr import SwinUNETR
from model.swinunetr_partial_v3 import SwinUNETR as SwinUNETR_partial_v3
from model.swinunetr_partial_onehot import SwinUNETR as SwinUNETR_partial_onehot
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key, organ_post_process

torch.multiprocessing.set_sharing_strategy('file_system')


def get_loader(args):
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ), # process h5 to here
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # ToTensord(keys=["image", "label", "post_label"]),
    ])
    pred_img = []
    pred_lbl = []
    pred_name = []
    for line in open(args.predict_data_txt_path):
        name = line.strip().split()[1].split('.')[0]
        pred_img.append(args.data_root_path + line.strip().split()[0])
        pred_lbl.append(args.data_root_path + line.strip().split()[1])
        pred_name.append(name)
    data_dicts = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(pred_img, pred_lbl, pred_name)]
    print('predict len {}'.format(len(data_dicts)))

    pred_dataset = Dataset(data=data_dicts, transform=val_transforms)
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    return pred_loader, val_transforms

def detransform_save(tensor_dict, input_transform, save_dir):
    post_transforms = Compose([
        Invertd(
            keys=['label', 'one_channel_pred'],
            transform=input_transform,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=True,
        ),
        # SaveImaged(
        #     keys=['label'],
        #     meta_keys='label_meta_dict',
        #     output_dir=save_dir,
        #     output_postfix='gt',
        #     resample=False,
        # ),
        SaveImaged(
            keys=['one_channel_pred'],
            meta_keys='label_meta_dict',
            output_dir=save_dir,
            output_postfix='pred',
            resample=False,
        ),
    ])
    return post_transforms(tensor_dict)

def predict(model, pred_loader, val_transforms, args):
    save_dir = os.path.join(
        args.log_dir, args.log_name, f'test_{os.path.basename(os.path.splitext(args.resume)[0])}', 'predict')
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()

    for index, batch in enumerate(tqdm(pred_loader)):
        image, label, name = batch["image"].cuda(), batch['label'], batch["name"]
        # if os.path.isfile(os.path.join(save_dir, name[0].split('/')[0], name[0].split('/')[-1] + '.npz')):
        if os.path.isdir(os.path.join(save_dir, name[0].split('/')[-1].split('.')[0])):
            continue
        with torch.no_grad():
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            if args.out_nonlinear == 'sigmoid':
                pred_sigmoid = F.sigmoid(pred)
                pred_hard = (pred_sigmoid > 0.5).cpu().numpy()
            elif args.out_nonlinear == 'softmax':
                c = pred.size(1)
                pred_hard = torch.argmax(pred, dim=1).cpu()
                pred_hard = F.one_hot(pred_hard, num_classes=c).permute(0, 4, 1, 2, 3)
                pred_hard = pred_hard[:, 1:]
                pred_hard = pred_hard.numpy()
        
        pred_hard_post = organ_post_process(pred_hard, args.organ_list)
        pred_hard_post = torch.tensor(pred_hard_post)
        torch.cuda.empty_cache()

        B, C, D, H, W = pred_hard_post.shape
        
        one_channel_pred = label.new_zeros((D, H, W))
        for icls in args.organ_list:
            one_channel_pred[pred_hard_post[0, icls-1] == 1] = icls

        batch['one_channel_pred'] = one_channel_pred.cpu()[None, None]
        de_batch = decollate_batch(batch)
        
        detransform_save(de_batch[0], val_transforms, os.path.join(save_dir))
        # os.makedirs(os.path.join(save_dir, name[0].split('/')[0]), exist_ok=True)
        # np.savez_compressed(
        #     os.path.join(save_dir, name[0].split('/')[0], name[0].split('/')[-1] + '.npz'),
        #     pred_sigmoid=pred_sigmoid[0].cpu().numpy())
        torch.cuda.empty_cache()
        del pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## logging
    parser.add_argument('--log_dir', default='output', help='Log directory.')
    parser.add_argument('--log_name', default='', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--model', type=str, choices=['swinunetr', 'swinunetr_partial', 'our_onehot'])
    parser.add_argument('--resume', default='', help='The path resume from checkpoint')
    # parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
    #                     help='The path of pretrain model')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--out_nonlinear', type=str, choices=['softmax', 'sigmoid'])
    parser.add_argument('--out_channels', type=int)
    ## dataset
    parser.add_argument('--data_root_path', default='', help='data root path')
    parser.add_argument('--predict_data_txt_path', type=str)
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
    parser.add_argument('--organ_list', nargs='+', type=int, required=True, help='Targget class ids for testing.')

    args = parser.parse_args()

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
            encoding=args.trans_encoding
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
    elif args.model == "our_onehot":
        model = SwinUNETR_partial_onehot(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            encoding=args.trans_encoding
        )

    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        load_dict, "module."
    )
    # args.epoch = checkpoint['epoch']

    # for key, value in load_dict.items():
    #     name = '.'.join(key.split('.')[1:])    # remove the 'module' prefix
    #     store_dict[name] = value

    model.load_state_dict(load_dict)
    print('Use pretrained weights')

    model.cuda()

    torch.backends.cudnn.benchmark = True

    pred_loader, pred_transforms = get_loader(args)

    predict(model, pred_loader, pred_transforms, args)

