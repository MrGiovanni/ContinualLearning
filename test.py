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
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete, Compose, Invertd, SaveImaged
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
# from monai.networks.nets import SwinUNETR

from model.swinunetr import SwinUNETR
from model.swinunetr_partial_v3 import SwinUNETR as SwinUNETR_partial_v3
from model.swinunetr_partial_onehot import SwinUNETR as SwinUNETR_partial_onehot
from dataset.dataloader_continue import get_loader
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ

torch.multiprocessing.set_sharing_strategy('file_system')

NUM_CLASS = 38

def detransform_save(tensor_dict, input_transform, save_dir):
    post_transforms = Compose([
        Invertd(
            keys=['label', 'one_channel_pred'],
            transform=input_transform,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=True,
        ),
        SaveImaged(
            keys=['label'],
            meta_keys='label_meta_dict',
            output_dir=save_dir,
            output_postfix='gt',
            resample=False,
        ),
        SaveImaged(
            keys=['one_channel_pred'],
            meta_keys='label_meta_dict',
            output_dir=save_dir,
            output_postfix='pred',
            resample=False,
        ),
    ])
    return post_transforms(tensor_dict)

def validation(model, ValLoader, val_transforms, args):
    save_dir = os.path.join(
        args.log_dir, args.log_name, f'test_{os.path.basename(os.path.splitext(args.resume)[0])}')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'predict'))
    
    model.eval()

    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    percase_result_str = ''
    
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        label[label == 255] = 0
        with torch.no_grad():
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            
            if args.out_nonlinear == 'sigmoid':
                pred_sigmoid = F.sigmoid(pred)
                #pred_hard = threshold_organ(pred_sigmoid, organ=args.threshold_organ, threshold=args.threshold)
                # pred_hard = threshold_organ(pred_sigmoid)
                pred_hard = pred_sigmoid > 0.5
            elif args.out_nonlinear == 'softmax':
                pred_hard = torch.argmax(pred, dim=1)
                pred_hard = F.one_hot(pred_hard, num_classes=args.out_channels).permute(0, 4, 1, 2, 3)
                pred_hard = pred_hard[:, 1:]
            # pred_hard = pred_hard.cpu().numpy()
        
        B, C, D, H, W = pred_hard.shape
        for b in range(B):
            case_name = name[b].split('/')[-1]
            content = f'case {case_name} | '
            template_key = get_key(name[b])
            # organ_list = TEMPLATE[template_key]
            # pred_hard_post = organ_post_process(pred_hard, args.organ_list)
            # pred_hard_post = torch.tensor(pred_hard_post)
            pred_hard_post = pred_hard

            for organ in args.organ_list:
                if torch.sum(label[b,organ-1,:,:,:].cuda()) != 0:
                    dice_organ, recall, precision = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,organ-1,:,:,:].cuda())
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item())
                    print('%s: dice %.4f, recall %.4f, precision %.4f.'%(ORGAN_NAME[organ-1], dice_organ.item(), recall.item(), precision.item()))
            print(content)
            percase_result_str += content + '\n'
        
        if args.store_result:
            # pred_sigmoid_store = (pred_sigmoid.cpu().numpy() * 255).astype(np.uint8)
            # label_store = (label.numpy()).astype(np.uint8)
            
            one_channel_pred = label.new_zeros((D, H, W))
            for icls in args.organ_list:
                one_channel_pred[pred_hard_post[0, icls-1] == 1] = icls
            # import pdb; pdb.set_trace()
            # np.savez_compressed(save_dir + '/predict/' + name[0].split('/')[0] + name[0].split('/')[-1], 
            #                 pred=pred_sigmoid_store, label=label_store)
            # pred_sigmoid_store = sitk.GetImageFromArray(pred_sigmoid_store)
            # label_store = sitk.GetImageFromArray(label_store)
            # sitk.WriteImage(pred_sigmoid_store, os.path.join(save_dir, 'predict', name[0].split('/')[-1] + 'pred.nii.gz'))
            # sitk.WriteImage(label_store, os.path.join(save_dir, 'predict', name[0].split('/')[-1] + 'label.nii.gz'))

            ### testing phase for this function
            # one_channel_label_v1, one_channel_label_v2 = merge_label(pred_hard_post, name)
            # batch['one_channel_label_v1'] = one_channel_label_v1.cpu()
            # batch['one_channel_label_v2'] = one_channel_label_v2.cpu()

            # _, split_label = merge_label(batch["post_label"], name)
            # batch['split_label'] = split_label.cpu()
            # print(batch['label'].shape, batch['one_channel_label'].shape)
            # print(torch.unique(batch['label']), torch.unique(batch['one_channel_label']))
            # visualize_label(batch, save_dir + '/output/' + name[0].split('/')[0] , val_transforms)
            ## load data
            # data = np.load('/out/epoch_80/predict/****.npz')
            # pred, label = data['pred'], data['label']

            batch['one_channel_pred'] = one_channel_pred.cpu()[None, None]
            de_batch = decollate_batch(batch)
            
            detransform_save(de_batch[0], val_transforms, os.path.join(save_dir, 'predict', name[0].split('/')[0]))
            
        del pred, pred_hard, pred_hard_post
        torch.cuda.empty_cache()
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open(os.path.join(save_dir, f'result.txt'), 'w') as f:
        for key in TEMPLATE.keys():
            # organ_list = TEMPLATE[key]
            content = 'Task%s| '%(key)
            for organ in args.organ_list:
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
            print(content)
            f.write(content)
            f.write('\n')
        content = 'Average | '
        for i in args.organ_list:
            content += '%s: %.4f, '%(ORGAN_NAME[i-1], ave_organ_dice[0][i-1] / ave_organ_dice[1][i-1])
        print(content)
        f.write(content)
        f.write('\n')
        print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        f.write('%s: %.4f, '%('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
        f.write('\n')
        f.write(percase_result_str)


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
    parser.add_argument('--log_name', default='', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--model', type=str, choices=['swinunetr', 'swinunetr_partial', 'our_onehot'])
    parser.add_argument('--resume', default='', help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--out_nonlinear', type=str, choices=['softmax', 'sigmoid'])
    parser.add_argument('--out_channels', type=int)
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', default='', help='data root path')
    parser.add_argument('--train_data_txt_path', type=str)
    parser.add_argument('--val_data_txt_path', type=str)
    parser.add_argument('--test_data_txt_path', type=str)
    parser.add_argument('--continue_data_txt_path', type=str)
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
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')

    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)
    
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
    
    # Load pre-trained weights
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        load_dict, "module."
    )

    model.load_state_dict(load_dict)
    print('Use pretrained weights')
    
    model.cuda()

    torch.backends.cudnn.benchmark = True

    test_loader, test_transforms = get_loader(args)

    validation(model, test_loader, test_transforms, args)

if __name__ == "__main__":
    main()
