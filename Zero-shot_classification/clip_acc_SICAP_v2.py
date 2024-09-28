import os
import random
import pandas as pd
from sklearn.metrics import classification_report
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

path = 'path/to/dataset/pathAGI/public_dataset/SICAPv2/raw/partition/Test/Train.xlsx'
data = pd.read_excel(path)
data = data.drop(columns=['G4C'])

label_list = ['NC', 'G3', 'G4', 'G5']
label_dict = {}
for idx, row in data.iterrows():
    img_name = row['image_name']
    labels = row.values[1:]
    if np.sum(labels) == 1:
        for label in labels:
            if label == 1:
                gt = label_list[np.where(labels == 1)[0][0]]
                if gt not in label_dict:
                    label_dict[gt] = []
                label_dict[gt].append(img_name)

data_2_label_dict = {}
img_list = []
for key in label_dict:
    random.shuffle(label_dict[key])
    for img in label_dict[key]:
        img = f'path/to/dataset/pathAGI/public_dataset/SICAPv2/raw/images/{img}'
        img_list.append(img)
        data_2_label_dict[img] = key
random.shuffle(img_list)

# CLIP-L
label_dict_map = {
    "NC": "Non-cancerous",
    "G3": "Gleason grade 3: Atrophic well differentiated and dense glandular regions",
    "G4": "Gleason grade 4: Cribriform, ill-formed, large-fused and papillary glandular patterns",
    "G5": "Gleason grade 5: Isolated cells or file of cells, nests of cells without lumina formation and pseudo-rosetting patterns"
}

label_dict = {
    "NC": 0,
    "G3": 1,
    "G4": 2,
    "G5": 3,
}

from config import ratio, epoch, mode, path, pretrained, model_type

if pretrained == 'openai':
    pretrained = 'openai'
else:
    pretrained = os.path.join(path, mode, 'ratio' + str(ratio), 'checkpoints', f'epoch_{epoch}.pt')
print(pretrained)

model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained,
                                                             cache_dir='path/to/model/open_clip', force_quick_gelu=True)

tokenizer = open_clip.get_tokenizer('ViT-B-16')
model = model.cuda()

text_label_list = ['An H&E image of {}'.format(label_dict_map[i].lower()) for i in label_dict.keys()]
text = tokenizer(text_label_list).cuda()

class PathologyDataset(Dataset):
    def __init__(self, img_list, data_2_label_dict, preprocess, label_dict):
        self.img_list = img_list
        self.data_2_label_dict = data_2_label_dict
        self.preprocess = preprocess
        self.label_dict = label_dict

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        label = self.label_dict[self.data_2_label_dict[img_path]]
        image = self.preprocess(image)
        return image, label

batch_size = 128
dataset = PathologyDataset(img_list, data_2_label_dict, preprocess, label_dict)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

cnt = 0
total = 0
gts = []
predicts = []

for images, labels in tqdm.tqdm(dataloader):
    images = images.cuda()
    labels = labels.cuda()
    total += labels.size(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    predict_labels = torch.argmax(text_probs, dim=-1)

    cnt += (predict_labels == labels).sum().item()
    gts.extend(labels.cpu().numpy())
    predicts.extend(predict_labels.cpu().numpy())

    if total % 100 == 0:
        print(cnt / total)

print(classification_report(gts, predicts, target_names=label_dict_map.values(), digits=3))
print('acc', cnt / total)
