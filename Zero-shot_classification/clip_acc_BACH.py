import os
import random
from sklearn.metrics import classification_report
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip
import numpy as np

label_dict = {
    "A": "Benign tissue",
    "B": "In-situ carcinoma",
    "C": "Invasive carcinoma",
    "D": "Normal tissue"
}

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = img_path.split('/')[-1].split('_')[-1][0]
        label = ord(label) - ord('A')
        image = self.preprocess(image)
        return image, label

path = 'path/to/public_data/BACH/ICIAR2018_BACH_Challenge/formalized_data/images'
images = [os.path.join(path, i) for i in os.listdir(path)]
from config import ratio, epoch, mode, path, pretrained, model_type

if pretrained == 'openai':
    pretrained = 'openai'
else:
    pretrained = os.path.join(path, mode, 'ratio'+str(ratio), 'checkpoints', f'epoch_{epoch}.pt')
print(pretrained)
model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained,
                                                       cache_dir='path/to/model/open_clip', force_quick_gelu=True)

keys = label_dict.keys()
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model = model.cuda()

text_label_list = ['An H&E image of {}'.format(label_dict[i].lower()) for i in keys]
text = tokenizer(text_label_list).cuda()

batch_size = 32  
dataset = ImageDataset(images, preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

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

print(classification_report(gts, predicts, target_names=label_dict.values(), digits=3))
print('acc', cnt / total)
