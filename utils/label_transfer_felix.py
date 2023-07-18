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

felix_label_mapping = {
    1: 8,    # Aorta
    2: 12,    # Adrenal Gland
    4: 33,    # CBD
    5: 25,    # Celiac abdominal aorta
    6: 18,    # Colon
    7: 14,    # Duodenum
    8: 4,    # Gall Bladder
    9: 9,    # IVC / postcava
    10: 3,    # L Kidney
    11: 2,    # R Kidney
    12: 6,    # Liver
    13: 11,    # Pancreas
    14: 34,    # pancreatic duct
    15: 35,    # SMA
    16: 19,    # small bowel / intestine
    18: 1,    # spleen
    19: 7,    # stomach
    20: 10,    # veins
    21: 36,    # Kidney_LtRV
    22: 37,    # Kidney_RtRV
    24: 38,    # CBD stent
}

right_left_organs = (
    (12, 13),
)

felix_label_full_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 25, 33, 34, 35, 36, 37, 38]


class ToTemplatelabeld(MapTransform):
    '''
    Comment: spleen to 1
    '''
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        label = data['label']
        # new_label = MetaTensor(torch.zeros_like(label), affine=label.affine, meta=label.meta, applied_operations=label.applied_operations)
        new_label = torch.zeros_like(label)
        for src, tgt in felix_label_mapping.items():
            new_label[label == src] = tgt
        data['label'] = new_label

        return data


class RL_Splitd(MapTransform):
    backend = ToTemplatelabeld.backend
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        label = data['label']
        # new_label = MetaTensor(torch.zeros_like(label), affine=label.affine, meta=label.meta, applied_operations=label.applied_operations)
        new_label = label.clone()
        for right_index, left_index in right_left_organs:
            new_label_np = self.rl_split(label[0].cpu().numpy(), right_index, left_index, data['name'])
            new_label_t = torch.as_tensor(new_label_np)
            print(new_label.shape, new_label_t.shape)
            new_label[new_label_t == left_index] = left_index
        data['label'] = new_label

        return data

    def rl_split(self, label, right_index, left_index, name):
        ccs = cc3d.connected_components(label == right_index, connectivity=26)
        values, counts = np.unique(ccs, return_counts=True)
        if len(values) > 3:
            sorted_index = sorted(range(len(counts)), key=counts.__getitem__, reverse=True)
            values_sorted = values[sorted_index]
            counts_sorted = counts[sorted_index]
            for c in values_sorted[3:]:
                ccs[ccs == c] = 0
            assert(values_sorted[0] == 0)
            ccs[ccs == values_sorted[1]] == 1
            ccs[ccs == values_sorted[2]] == 2

            print(f'In {name}, throw {values_sorted[3:]} small regions with {sum(counts_sorted[3:])} voxels.')

        xs1, _, _ = np.nonzero(ccs == 1)
        xs2, _, _ = np.nonzero(ccs == 2)

        new_label = label.copy()
        if len(xs1) == 0:
            return new_label[None]
        if np.mean(xs1) < np.mean(xs2):
            new_label[ccs == 1] = left_index
            new_label[ccs == 2] = right_index
        else:
            new_label[ccs == 2] = left_index
            new_label[ccs == 1] = right_index
        return new_label[None]


def convert_to_onehot_label(label, raw_label, name):
    shape = tuple([label.shape[0], num_class, *label.shape[2:]])
    result = torch.zeros(shape)

    batch_size = result.shape[0]
    for b in range(batch_size):
        # -1 for organ not labeled
        for c in range(1, num_class + 1):
            if c not in felix_label_full_list:
                result[b, c-1] = -1
            else:
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

    y = convert_to_onehot_label(y, y_raw, name)
    store_y = y.numpy().astype(np.uint8)
    with h5py.File(os.path.join(post_label_dir, f'{basename}.h5'), 'w') as f:
        f.create_dataset('post_label', data=store_y, compression='gzip', compression_opts=9)
        f.close()


if __name__ == '__main__':
    data_dir = './data'
    data_list = 'dataset/dataset_list/2felix_5mm_step1_ours_ilt_new.txt'
    post_label_dir = '/ccvl/net/ccvl15/yixiao/CLIP-Driven-Universal-Model/data/16_FELIX_lowres/2post_label_5mm_step1_ours_ilt_new'
    # data_dir = '/mnt/sdi/yixiao/med_seg_data'
    # data_list = 'dataset/dataset_list/felix_small_step2_train.txt'
    # post_label_dir = '/mnt/sdi/yixiao/med_seg_data/16_FELIX_lowres/post_label_small_4mm'
    os.makedirs(post_label_dir, exist_ok=True)

    dataset_input = list()
    with open(data_list) as f:
        for line in f:
            image_path, label_path = line.split()
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
            ToTemplatelabeld(keys=['label']),
            RL_Splitd(keys=['label']),
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
