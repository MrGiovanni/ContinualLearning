import os
import clip
import torch


## PAOT
ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 
                'Colon Tumor', 'Kidney Cyst']

organ_name_low = ['spleen', 'kidney_right', 'kidney_left', 'gall_bladder', 'esophagus', 
                'liver', 'stomach', 'aorta', 'postcava', 'portal_vein_and_splenic_vein',
                'pancreas', 'adrenal_gland_right', 'adrenal_gland_left', 'duodenum', 'hepatic_vessel',
                'lung_right', 'lung_left', 'colon', 'intestine', 'rectum', 
                'bladder', 'prostate', 'femur_left', 'femur_right', 'celiac_truck',
                'kidney_tumor', 'liver_tumor', 'pancreas_tumor', 'hepatic_vessel_tumor', 'lung_tumor', 
                'colon_tumor', 'kidney_cyst']

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

for idx,organ in enumerate(ORGAN_NAME):
    text_inputs = clip.tokenize(f'A computerized tomography of a {organ}').to(device)
    
    # Calculate text embedding features
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        print(text_features.shape, text_features.dtype)
        torch.save(text_features, organ_name_low[idx]+'.pth')

