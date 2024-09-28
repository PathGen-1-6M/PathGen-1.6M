import os
import random
import re
from sklearn.metrics import classification_report
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = 'path/to/public_data/Osteosarcoma Tumor Assessment/ML_Features_1144.csv'
data = pd.read_csv(path)
cats_dict = {
    "Non-tumor": "Non-tumor",
    "Necrotic tumor": "Necrotic tumor",
    "Viable tumor": "Viable tumor"
}
label_dict = {
    'Non-tumor': 0,
    'Necrotic tumor': 1,
    'Viable tumor': 2
}
csv2_label_dict = {
    "Non-Tumor": "Non-tumor",
    "Viable": "Viable tumor",
    "viable: non-viable": "Viable tumor",
    "Non-Viable-Tumor": "Necrotic tumor",
}

img_list = []
img_2_label_dict = {}
for idx, row in data.iterrows():
    img_name = row['image.name']
    img_name = re.sub(' +', '-', img_name)
    img_name = re.sub('-+', '-', img_name) + '.jpg'
    gt = csv2_label_dict[row['classification']]
    img_path = f'path/to/project/2023/pathAGI/PathAGI_tool_lab/construct_public_gathered_data/data/osteo/images/{img_name}'
    img_list.append(img_path)
    img_2_label_dict[img_path] = gt

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


text_label_list = ['An H&E image patch of {}'.format(cats_dict[i].lower()) for i in label_dict.keys()]
text = tokenizer(text_label_list).cuda()


class PathologyDataset(Dataset):
    def __init__(self, img_list, img_2_label_dict, preprocess, label_dict):
        self.img_list = img_list
        self.img_2_label_dict = img_2_label_dict
        self.preprocess = preprocess
        self.label_dict = label_dict

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        label = self.label_dict[self.img_2_label_dict[img_path]]
        image = self.preprocess(image)
        return image, label


batch_size = 128
dataset = PathologyDataset(img_list, img_2_label_dict, preprocess, label_dict)
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


print(classification_report(gts, predicts, target_names=cats_dict.values(), digits=3))
print('acc', cnt / total)
