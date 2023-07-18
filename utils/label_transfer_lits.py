import cc3d
import h5py
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import MetaTensor, DataLoader, Dataset, list_data_collate
from monai import transforms
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from multiprocessing import Pool
import numpy as np
import os
import SimpleITK as sitk
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

torch.multiprocessing.set_sharing_strategy('file_system')

num_class = 38


def convert_to_onehot_label(label):
    shape = tuple([label.shape[0], num_class, *label.shape[2:]])
    result = torch.zeros(shape)

    batch_size = result.shape[0]
    for b in range(batch_size):
        for c in range(1, num_class + 1):
            result[b, c-1] = (label[b, 0] == c)

    return result


def process_single_case(batch):
    x, y, y_raw, name = batch["image"], batch["label"], batch['label_raw'], batch['name']
    basename = os.path.basename(name[0]).split('.')[0]
    if os.path.isfile(os.path.join(post_label_dir, f'{basename}.h5')):
        return
    print(name)

    # import pdb; pdb.set_trace()
    # image = x.cpu().numpy()
    # image = sitk.GetImageFromArray(image)
    # sitk.WriteImage(image, f'{name[0]}')
    # post_label = y.cpu().numpy()
    # post_label = sitk.GetImageFromArray(post_label)
    # sitk.WriteImage(post_label, f'{name[0][:-7]}_post_label.nii.gz')

    y = convert_to_onehot_label(y)
    store_y = y.numpy().astype(np.uint8)
    with h5py.File(os.path.join(post_label_dir, f'{basename}.h5'), 'w') as f:
        f.create_dataset('post_label', data=store_y, compression='gzip', compression_opts=9)
        f.close()


if __name__ == '__main__':
    data_dir = './data/'
    data_list = 'dataset/dataset_list/lits_train_pseudo_btcv_onehot.txt'
    post_label_dir = './data/04_LiTS/pseudo_post_label_onehot'
    os.makedirs(post_label_dir, exist_ok=True)

    dataset_input = list()
    with open(data_list) as f:
        for line in f:
            image_path, label_path = line.split()
            fname = image_path.split('/')[-1]
            basename = fname.split('.')[0]
            if os.path.isfile(os.path.join(post_label_dir, f'{basename}.h5')):
                continue
            dataset_input.append({
                'image': os.path.join(data_dir, image_path),
                'label': os.path.join(data_dir, label_path),
                'label_raw': os.path.join(data_dir, label_path),
                'name': os.path.basename(label_path)
            })
    print(f'len of dataset {len(dataset_input)}')

    label_process = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label", "label_raw"], image_only=True),
            transforms.AddChanneld(keys=["image", "label", "label_raw"]),
            transforms.Orientationd(keys=["image", "label", "label_raw"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label", "label_raw"],
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "nearest", "nearest"),), # process h5 to here
        ]
    )

    dataset = Dataset(data=dataset_input, transform=label_process)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)

    # with Pool(10) as p:
    #     for res in p.imap(process_single_case, dataloader):
    #         pass
    # p.close()
    for batch in dataloader:
        process_single_case(batch)
