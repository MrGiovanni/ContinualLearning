# Continual Learning

## Paper

<b>Continual Learning for Abdominal Multi-Organ and Tumor Segmentation</b> <br/>
[Yixiao Zhang](https://scholar.google.com/citations?user=lU3wroMAAAAJ), [Xinyi Li](https://www.linkedin.com/in/lixinyi808/), [Huimiao Chen](https://scholar.google.com/citations?hl=en&user=o-F4kBgAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Yaoyao Liu](https://www.cs.jhu.edu/~yyliu/), and [Zongwei Zhou](https://www.zongweiz.com/) <br/>
Johns Hopkins University  <br/>
MICCAI 2023 (early accept) <br/>
[paper](https://arxiv.org/pdf/2306.00988.pdf) | [code](https://github.com/MrGiovanni/ContinualLearning)

## 0. Installation

```bash
git clone https://github.com/MrGiovanni/ContinualLearning
```
See [installation instructions](document/INSTALL.md) to create an environment and obtain requirements.

## 1. Prepare your datasets
#### 1.1 Prepare your image and label files in a customer path, then create a txt file in the dataset/dataset_list folder. See txt files under dataset/dataset_list folder as examples.
#### 1.2 Put the class name embedding file word_embedding_38class.pth under ./pretrained_weights/ folder.

## 2. Training
Use the train.py file for training models. An example script is
```
python train.py 
--phase train
--data_root_path ./data
--train_data_txt_path ./dataset/dataset_list/btcv_train.txt
--val_data_txt_path ./dataset/dataset_list/btcv_val.txt
--organ_list 1 2 3 4 5 6
--max_epoch 101
--warmup_epoch 15
--batch_size 2
--num_samples 1
--lr 1e-4
--model swinunetr
--trans_encoding word_embedding
--word_embedding ./pretrained_weights/word_embedding_38class.pth
--out_nonlinear softmax
--out_channels 38
--log_name your_log_folder_name
```
Switch the argument `--model` for different models: `swinunetr` for SwinUNETR, `swinunetr_partial` for the proposed model with organ-specific segmentation heads (this model should be used with `--out_nonlinear sigmoid`).

## 3. Testing
Use the test.py file for testing models. An example script is
```
python test.py
--log_name your_log_folder_name
--resume your_checkpoint_path
--data_root_path ./data
--test_data_txt_path ./dataset/dataset_list/btcv_test.txt
--organ_list 1 2 3 4 5 6
--model swinunetr
--out_nonlinear softmax
--out_channels 38
```

## 4. Predicting and Pseudo Label Preparing
Use predict.py to make organ segmentation predictions on the test images or images for continual learning
```
python predict.py \
    --log_name your_log_folder_name \
    --resume your_checkpoint_path \
    --data_root_path ./data \
    --predict_data_txt_path ./dataset/dataset_list/<step2 train files>.txt \
    --organ_list 1 2 3 4 5 6 \
    --model swinunetr \
    --out_nonlinear softmax \
    --out_channels 38
```
You would need to combine the predicted segmentation on some organs together with ground truth segmentation on other organs to generate pseudo labels for continual learning. Please take scripts in prepare_pseudo_dataset.ipynb as examples.

## 5. Continual learning
For the proposed model, in continual learning stages, the same train.py is used to train the model but with different settings.
```
python train.py \
  --phase continue \
  --data_root_path ./data \
  --continue_data_txt_path ./dataset/dataset_list/<data list with generated pseudo label>.txt \
  --organ_list 1 2 3 4 5 6 7 8 9 10 11 12 \
  --num_workers 8 \
  --max_epoch 101 \
  --warmup_epoch 15 \
  --store_num 10 \
  --batch_size 2 \
  --num_samples 1 \
  --lr 1e-4 \
  --model swinunetr_partial \
  --out_nonlinear sigmoid \
  --out_channels 38 \
  --pretrain checkpoint_of_previous_learning_stage \
  --trans_encoding word_embedding \
  --word_embedding ./pretrained_weights/word_embedding_38class.pth \
  --log_name your_log_folder_name
```
The differences are that you are now using pseudo labels listed in the `continue_data_txt_path` file, training on both pretrained and continual organ targets, and use a checkpoint from the previous learning stage.

## Acknowledgements
This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and partially by the Patrick J. McGovern Foundation Award. We appreciate the effort of the MONAI Team to provide open-source code for the community.
