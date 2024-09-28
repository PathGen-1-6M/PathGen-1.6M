import os
import random
import pandas as pd
from sklearn.metrics import classification_report
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

path = 'path/to/dataset/pathAGI/public_dataset/skincancer/raw/tiles-v2.csv'
data = pd.read_csv(path)

label_dict = {
    "nontumor_skin_chondraltissue_chondraltissue": "Non-tumor chondral tissue",
    "nontumor_skin_dermis_dermis": "Non-tumor dermis",
    "nontumor_skin_elastosis_elastosis": "Non-tumor elastosis",
    "nontumor_skin_epidermis_epidermis": "Non-tumor epidermis",
    "nontumor_skin_hairfollicle_hairfollicle": "Non-tumor hair follicle",
    "nontumor_skin_muscle_skeletal": "Non-tumor skeletal muscle",
    "nontumor_skin_necrosis_necrosis": "Non-tumor necrosis",
    "nontumor_skin_nerves_nerves": "Non-tumor nerves",
    "nontumor_skin_sebaceousglands_sebaceousglands": "Non-tumor sebaceous glands",
    "nontumor_skin_subcutis_subcutis": "Non-tumor subcutis",
    "nontumor_skin_sweatglands_sweatglands": "Non-tumor sweat glands",
    "nontumor_skin_vessel_vessel": "Non-tumor vessel",
    "tumor_skin_epithelial_bcc": "Tumor epithelial basal cell carcinoma",
    "tumor_skin_epithelial_sqcc": "Tumor epithelial squamous cell carcinoma",
    "tumor_skin_melanoma_melanoma": "Tumor melanoma",
    "tumor_skin_naevus_naevus": "Tumor naevus"
}

label_index_dict = {
    "nontumor_skin_chondraltissue_chondraltissue": 0,
    "nontumor_skin_dermis_dermis": 1,
    "nontumor_skin_elastosis_elastosis": 2,
    "nontumor_skin_epidermis_epidermis": 3,
    "nontumor_skin_hairfollicle_hairfollicle": 4,
    "nontumor_skin_muscle_skeletal": 5,
    "nontumor_skin_necrosis_necrosis": 6,
    "nontumor_skin_nerves_nerves": 7,
    "nontumor_skin_sebaceousglands_sebaceousglands": 8,
    "nontumor_skin_subcutis_subcutis": 9,
    "nontumor_skin_sweatglands_sweatglands": 10,
    "nontumor_skin_vessel_vessel": 11,
    "tumor_skin_epithelial_bcc": 12,
    "tumor_skin_epithelial_sqcc": 13,
    "tumor_skin_melanoma_melanoma": 14,
    "tumor_skin_naevus_naevus": 15
}

img_list = []
data_2_label_dict = {}
for idx, row in data.iterrows():
    img_name = row['file'].replace('data/', '')
    gt = row['class']
    img = f'path/to/dataset/pathAGI/public_dataset/skincancer/raw/{img_name}'
    img_list.append(img)
    data_2_label_dict[img] = gt

random.shuffle(img_list)

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

text_label_list = ['An H&E image of {}'.format(label_dict[i].lower()) for i in label_dict.keys()]
text = tokenizer(text_label_list).cuda()

class PathologyDataset(Dataset):
    def __init__(self, img_list, data_2_label_dict, preprocess, label_index_dict):
        self.img_list = img_list
        self.data_2_label_dict = data_2_label_dict
        self.preprocess = preprocess
        self.label_index_dict = label_index_dict

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        label = self.label_index_dict[self.data_2_label_dict[img_path]]
        image = self.preprocess(image)
        return image, label

batch_size = 128
dataset = PathologyDataset(img_list, data_2_label_dict, preprocess, label_index_dict)
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

print(classification_report(gts, predicts, target_names=[label_dict[key] for key in label_index_dict.keys()], digits=3))
print('acc', cnt / total)
