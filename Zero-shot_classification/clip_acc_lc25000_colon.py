import os
import random
from sklearn.metrics import classification_report
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_dict = {
    'colon_n': 'normal colon tissue', 'colon_aca': 'colon adenocarcinoma'
}
label_index_dict = {
    'colon_n': 0, 'colon_aca': 1
}

class PathologyDataset(Dataset):
    def __init__(self, img_list, label_index_dict, preprocess):
        self.img_list = img_list
        self.label_index_dict = label_index_dict
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        label = self.label_index_dict[os.path.basename(os.path.dirname(img_path))]
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

text_label_list = ['An image of {}'.format(label_dict[i].lower()) for i in label_dict.keys()]
text = tokenizer(text_label_list).cuda()

path = 'path/to/dataset/pathasst_data/clip_zeroshot_cls/colon_image_sets'

def walk_dir(data_dir, file_types=['.jpeg']):
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            if any(f.lower().endswith(ft) for ft in file_types):
                path_list.append(os.path.join(dirpath, f))
    return path_list

img_list = walk_dir(path, ['.jpeg'])
random.shuffle(img_list)

batch_size = 128
dataset = PathologyDataset(img_list, label_index_dict, preprocess)
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

print(classification_report(gts, predicts, target_names=label_dict.values(), digits=3))
print('acc', cnt / total)
