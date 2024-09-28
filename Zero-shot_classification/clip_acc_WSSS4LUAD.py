import os
import random
from sklearn.metrics import classification_report
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

label_dict = {
    "tumor": 0,
    "normal": 1
}

class PathologyDataset(Dataset):
    def __init__(self, img_list, preprocess, label_dict):
        self.img_list = img_list
        self.preprocess = preprocess
        self.label_dict = label_dict

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        label_info = eval(img_path.split('/')[-1].split('-')[-1][:-4])
        label = 0 if label_info[0] == 1 else 1
        image = self.preprocess(image)
        return image, label

from config import ratio, epoch, mode, path, pretrained, model_type

if pretrained == 'openai':
    pretrained = 'openai'
else:
    pretrained = os.path.join(path, mode, 'ratio' + str(ratio), 'checkpoints', f'epoch_{epoch}.pt')
print(pretrained)

model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained,
                                                             cache_dir='path/to/model/open_clip', force_quick_gelu=True)

# 获取tokenizer
tokenizer = open_clip.get_tokenizer('ViT-L-14')
model = model.cuda()

text_label_list = ['An H&E image of tumor tissue.', 'An H&E image of normal tissue.']
text = tokenizer(text_label_list).cuda()

path = 'path/to/dataset/pathasst_data/clip_zeroshot_cls/WSSS/1.training'
img_list = [os.path.join(path, i) for i in os.listdir(path) if i.endswith('.png')]
random.shuffle(img_list)

batch_size = 128
dataset = PathologyDataset(img_list, preprocess, label_dict)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

cnt = 0
total = 0
gts = []
predicts = []
img_dict = {'tumor': 0, 'normal': 0}

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

    for label in labels.cpu().numpy():
        if label == 0:
            img_dict['tumor'] += 1
        else:
            img_dict['normal'] += 1

    if total % 100 == 0:
        print(cnt / total)

print(img_dict)
print(classification_report(gts, predicts, target_names=['tumor', 'normal'], digits=3))
print('acc', cnt / total)
