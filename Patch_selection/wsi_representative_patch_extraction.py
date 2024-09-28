# Imports
import os
import sys
import json
import math
import torch
import random
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from deepslide.dispatch import openSlide
from build_data import ov_patch_dataset_1wsi
from region.tumor_region_segmentation_old import seg_tissue
from .utils import wsi_dict
import open_clip
import cv2

# Configuration
sys.path.append("path/to/project/2024/WSI_Captioning/patch_implement_lab")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Constants
kmeans_ratio = 1.6
device = 6
ds = 32  # Downsampling 32x
level = 0  # Patch crop level
scale = 2 ** level  # Image scale
patch_size = 336 * 2
tiles_size = (336 * 8 * (2 ** level), 336 * 8 * (2 ** level))  # 256 instances = 1 tile
ref_tile_size = (int(tiles_size[0] / ds), int(tiles_size[1] / ds))  # Tile shape in downsampling
ref_size = int(patch_size * scale / ds)  # Instance shape in downsampling

# Font and model setup
font = ImageFont.truetype('path/to/project/2024/WSI_Captioning/patch_implement_lab/NotoSansCJK-Regular.ttc', 30)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14-336',
    pretrained='path/to/model/clip/exp/ViT-L-14-336/240103/pathclip.pt',
    cache_dir='path/to/model/open_clip',
    force_quick_gelu=True
)
model = model.cuda(device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14-336')


# Helper functions
def draw_imgs(img, target_wsi_path, patch_loc, indices, mode='direct'):
    imga = img.convert('RGBA')
    img1 = imga.copy()
    transp = Image.new('RGBA', imga.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(transp, 'RGBA')

    for idx, prob_inds in enumerate(indices[:64]):
        loc = patch_loc[prob_inds]
        r, c = int(loc[0] / ds), int(loc[1] / ds)
        color = 'yellow' if idx < 6 else 'red'
        draw.text([r + 5, c + 5], str(idx), font=font, fill=(0, 0, 0, 255))
        draw.rectangle([r, c, r + ref_size - 1, c + ref_size - 1], fill=color, outline=None)

    img1.paste(Image.blend(img1, transp, alpha=0.5))
    plt.figure(figsize=(20, 20))
    plt.imshow(img1)
    plt.savefig(os.path.join(target_wsi_path, f'topk_{mode}.png'))


def kmeans_cluster(k, target_wsi_path, img, img_features_list, img_locations):
    img_features_list = img_features_list / np.linalg.norm(img_features_list, axis=1, keepdims=True)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_features_list)
    silhouette = silhouette_score(img_features_list, kmeans.labels_)
    dbi = davies_bouldin_score(img_features_list, kmeans.labels_)
    chi = calinski_harabasz_score(img_features_list, kmeans.labels_)

    # Draw clusters
    labels = kmeans.labels_
    colors = [
        'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta',
        'lime', 'olive', 'chocolate', 'gold', 'salmon', 'navy', 'teal', 'maroon', 'coral',
        'lightblue', 'darkgreen', 'tan', 'silver', 'darkred'
    ]
    img_draw = img.convert('RGBA')
    draw = ImageDraw.Draw(img_draw, 'RGBA')

    for label, loc in zip(labels, img_locations):
        r, c = int(loc[0] / ds), int(loc[1] / ds)
        color = colors[label % len(colors)]
        draw.rectangle([r, c, r + ref_size - 1, c + ref_size - 1], fill=color, outline=None)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_draw)
    plt.savefig(os.path.join(target_wsi_path,
                             f'optimal_k_kmeans_{k}_{round(silhouette, 2)}_{round(dbi, 2)}_{round(chi, 2)}.png'))

    # Select images
    clusters = {i: [] for i in range(k)}
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    selected_images = []
    avg_per_cluster = 256 / k
    for label, cluster_indices in clusters.items():
        count = max(1, int(avg_per_cluster))
        selected_indices = cluster_indices if len(cluster_indices) < avg_per_cluster else random.sample(cluster_indices,
                                                                                                        count)
        selected_images.extend([(img_locations[idx], label) for idx in selected_indices])
        if len(selected_images) >= 256:
            break
    return selected_images

valid_wsi_data = 'svs_path_list.txt'
with open(valid_wsi_data, 'r') as f:
    data = f.readlines()
data = [x.strip().replace('.svs', '') for x in data]
import pdb

pdb.set_trace()
cls = [x.split('/')[3] for x in data]
cls_dict = {data[i].split('/')[-1]: cls[i] for i in range(len(data))}
def extract_topk_and_feature(wsi_path, cap_dict, target_wsi_path):
    text_list = wsi_dict[cls_dict[wsi_path]]
    for k, v in cap_dict.items():
        text_list.append(v)

    # Load WSI image
    wsi_obj = openSlide(wsi_path)
    img = Image.fromarray(wsi_obj.read(scale=ds))
    width, height = wsi_obj.width, wsi_obj.height
    print(width, height)

    # Segment tissue regions
    tissue_region, _, _ = seg_tissue(img)
    labeled_region = skimage.measure.label(tissue_region)
    region_list = skimage.measure.regionprops(labeled_region)
    patch_cor_list = []
    overlap = 1

    for ri in region_list:
        min_row, min_col, max_row, max_col = ri.bbox
        for xi in np.arange(min_col, max_col, int(patch_size / round(ds) * overlap)):
            for yi in np.arange(min_row, max_row, int(patch_size / round(ds) * overlap)):
                p_region = tissue_region[yi:yi + ref_size, xi:xi + ref_size]
                if p_region.sum() / p_region.size > 0.5:
                    x_ori, y_ori = round(xi * ds), round(yi * ds)
                    patch_cor_list.append([x_ori, y_ori, xi, yi])

    # Load dataset and model
    wsi_dataset = ov_patch_dataset_1wsi(wsi_obj, patch_cor_list, ds, patch_size, resize=336, return_ori=True)
    loader = torch.utils.data.DataLoader(dataset=wsi_dataset, batch_size=128, shuffle=False, num_workers=10)
    text_label_list = [f'{i}' for i in text_list]

    patch_loc, img_features_list = [], []
    with torch.no_grad():
        text = tokenizer(text_label_list).cuda(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Extract image features
        for step, (imgs, xi, yi) in tqdm(enumerate(loader), total=len(loader)):
            imgs = imgs.cuda(device).squeeze(0)
            image_features = model.encode_image(imgs, True)
            img_features_list.append(image_features.cpu().numpy())
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = 100.0 * image_features @ text_features.T
            out = text_probs.data.cpu()

            if step == 0:
                features_3584 = out
            else:
                features_3584 = torch.cat((features_3584, out), dim=0)

            patch_loc.extend(np.vstack((xi.data.numpy(), yi.data.numpy())).T.tolist())

    # Sort and cluster patches
    values, indices = torch.sort(torch.max(features_3584[:, :20], dim=1).values, descending=True)
    values2, indices2 = torch.sort(torch.sum(features_3584[:, 20:], dim=-1), descending=True)
    img_features_list = np.concatenate(img_features_list, axis=0)
    selected_images = kmeans_cluster(math.ceil(math.sqrt(len(loader) * kmeans_ratio)), target_wsi_path, img,
                                     img_features_list, patch_loc)

    # Draw images
    draw_imgs(img, target_wsi_path, patch_loc, indices, mode='direct')
    draw_imgs(img, target_wsi_path, patch_loc, indices2, mode='caption')

    return [], [], [], img_features_list, patch_cor_list


# Main execution
wsis_path = 'svs_path_list.txt'
cap_path = 'wsi_caption_dict.json'
target_root = 'path/to/dataset/wsi_captioning/dataset/WSI_patch_extraction'
os.makedirs(target_root, exist_ok=True)
STEP = 50
bag = 2

with open(wsis_path, 'r') as f:
    wsi_list = [wsi.strip() for wsi in f.readlines() if wsi.strip()]

with open(cap_path, 'r') as f:
    cap_dict = json.load(f)

for wsi_idx, wsi_path in enumerate(wsi_list[bag * STEP:(bag + 1) * STEP]):
    print(f"{wsi_idx} / {len(wsi_list[bag * STEP:(bag + 1) * STEP])} {wsi_path}")
    wsis_name = wsi_path.split('/')[-1].replace('.svs', '')
    target_wsi_path = os.path.join(target_root, wsis_name)
    target_wsi_img_path = os.path.join(target_wsi_path, 'imgs_direct_prompt')
    target_wsi_img_path2 = os.path.join(target_wsi_path, 'imgs_caption_prompt')
    target_wsi_img_path3 = os.path.join(target_wsi_path, 'imgs_kmeans')

    os.makedirs(target_wsi_path, exist_ok=True)
    os.makedirs(target_wsi_img_path, exist_ok=True)
    os.makedirs(target_wsi_img_path2, exist_ok=True)
    os.makedirs(target_wsi_img_path3, exist_ok=True)

    try:
        cap = cap_dict[wsis_name]['refined']
        extracted_patches, extracted_patches2, extracted_patches3, img_features_list, patch_cor_list = extract_topk_and_feature(
            wsi_path, cap, target_wsi_path)
        torch.cuda.empty_cache()
    except Exception as e:
        print(e)
        continue

    np.save(os.path.join(target_wsi_path, 'img_features.npy'), img_features_list)
    np.save(os.path.join(target_wsi_path, 'patch_cor_list.npy'), patch_cor_list)

    # Save extracted patches
    for idx, (patch, loc_x, loc_y) in enumerate(extracted_patches):
        Image.fromarray(patch).save(os.path.join(target_wsi_img_path, f'{idx}_{loc_x}_{loc_y}.jpg'))

    for idx, (patch, loc_x, loc_y) in enumerate(extracted_patches2):
        Image.fromarray(patch).save(os.path.join(target_wsi_img_path2, f'{idx}_{loc_x}_{loc_y}.jpg'))

    for idx, (patch, loc_x, loc_y, label) in enumerate(extracted_patches3):
        Image.fromarray(patch).save(os.path.join(target_wsi_img_path3, f'{loc_x}_{loc_y}_{label}.jpg'))
