# PathGen-1.6M

This is the official repo for  **PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration**

[**Dataset**](https://github.com/PathGen-1-6M/PathGen-1.6M/tree/main/Data)

## Abstract

Vision Language Models (VLMs) like CLIP have attracted substantial attention in pathology, serving as backbones for applications such as zero-shot image classification and Whole Slide Image (WSI) analysis. Additionally, they can function as vision encoders when combined with large language models (LLMs) to support broader capabilities. Current efforts to train pathology VLMs rely on pathology image-text pairs from platforms like PubMed, YouTube, and Twitter, which provide limited, unscalable data with generally suboptimal image quality. In this work, we leverage large-scale WSI datasets like TCGA to extract numerous high-quality image patches. We then train a large multimodal model to generate captions for these images, creating PathGen-1.6M, a dataset containing 1.6 million high-quality image-caption pairs. Our approach involves multiple agent models collaborating to extract representative WSI patches, generating and refining captions to obtain high-quality image-text pairs. Extensive experiments show that integrating these generated pairs with existing datasets to train a pathology-specific CLIP model, PathGen-CLIP, significantly enhances its ability to analyze pathological images, with substantial improvements across nine pathology-related zero-shot image classification tasks and three whole-slide image tasks. Furthermore, we construct 200K instruction-tuning data based on PathGen-1.6M and integrate PathGen-CLIP with the Vicuna LLM to create more powerful multimodal models through instruction tuning. Overall, we provide a scalable pathway for high-quality data generation in pathology, paving the way for next-generation general pathology models.

![image-20240817142901198](./imgs/performance_overview.png)

## Method

![image-20240817142637866](./imgs/method.png)

We employ multiple agents working collaboratively to generate high-quality pathology image-text pairs. This process involves extracting representative WSI image patches by generating text prompts for CLIP to retrieve the most relevant patches. These patches are then described by a trained pathology LMM agent, followed by another LMM agent that revises and summarizes the descriptions. 

## Usage of PathGen-1.6M Dataset

Download the PathGen Dataset:

> #### Step1:

Access and download the JSON file containing image names, specific positions, and captions from the [**Dataset**](https://github.com/PathGen-1-6M/PathGen-1.6M/tree/main/Data).This file is critical for the subsequent steps as it provides the necessary metadata.

> #### Step2:

Employ the GDC Data Transfer Tool to download the whole-slide images (.svs files) referenced in the JSON file. Detailed instructions for using this tool can be found on the GDC's documentation page: https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/.

> #### Step3:

Follow the following code to gather image-caption pairs.

```python
import os
import json
from PIL import Image
import openslide

# Define paths and configuration
WSI_DIR = "/path/to/your/wsi/files"  # Update this to the directory containing your WSI files
OUTPUT_DIR = "./output"  # Directory where patches and captions will be saved
PATCH_SIZE = (672, 672)  # Size of the patch to extract

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the list of WSIs with positions and captions
# data = [
#     {
#         "WSI_id": "TCGA-22-5474-01Z-00-DX1.8736FB24-7E65-4ACB-9325-382D7F864F62",
#         "position": ["41024", "35104"],
#         "caption": "The tissue image reveals dense cellular infiltration, suggesting inflammation, and cells with large, hyperchromatic nuclei and high nuclear-to-cytoplasmic ratios indicative of a neoplastic process. Pink, acellular material points to fibrosis or connective tissue. The disrupted architecture further supports a pathological condition, possibly cancer combined with fibrotic changes."
#     },
#     {
#         "WSI_id": "TCGA-55-8621-01Z-00-DX1.7C519007-D59D-432A-BF4D-23D14A1C8BB6",
#         "position": ["13280", "13056"],
#         "caption": "The lung tissue image displays myofibroblasts with elongated nuclei and eosinophilic cytoplasm, indicative of collagen-rich fibrosis. Epithelial cells, forming glandular structures, show cellular atypia. The architecture is disrupted by dense fibrotic areas and patchy cellular infiltration, suggesting an interstitial lung disease characterized by chronic fibrosis and inflammation. Hemorrhage or hemosiderin deposits are not evident."
#     },
#     {
#         "WSI_id": "TCGA-AH-6547-01Z-00-DX1.73040c3e-8219-4d21-88f2-613218d32297",
#         "position": ["5472", "8320"],
#         "caption": "The tissue shows irregular, atypical glandular structures indicative of adenocarcinoma, with hyperchromatic nuclei, high nuclear-to-cytoplasmic ratio, and pleomorphism. Desmoplastic stroma and mitotic figures suggest high-grade dysplasia. These features confirm a diagnosis of malignant adenocarcinoma of the rectum, characterized by loss of normal glandular architecture and cellular disorganization."
#     }
# ]
pathgen_data_path = 'PathGen_dataset.json'
with open(pathgen_data_path, 'r') as f:
    data = json.load(f)

def extract_patch_from_wsi(wsi_path, position, patch_size):
    """
    Extracts a patch from the WSI at the specified position.

    :param wsi_path: Path to the WSI file.
    :param position: Tuple of (x, y) coordinates.
    :param patch_size: Size of the patch to extract.
    :return: Extracted patch as a PIL Image.
    """
    try:
        # Load WSI using OpenSlide
        wsi = openslide.OpenSlide(wsi_path)
        x, y = map(int, position)  # Convert position coordinates to integers
        patch = wsi.read_region((x, y), 0, patch_size)  # Extract patch
        return patch
    except Exception as e:
        print(f"Error extracting patch from {wsi_path} at {position}: {e}")
        return None


# Process each WSI and its corresponding data
for item in data:
    wsi_id = item['WSI_id']
    position = item['position']
    caption = item['caption']

    # Construct the full path to the WSI file
    wsi_path = os.path.join(WSI_DIR, f"{wsi_id}.svs")  # Update extension if different

    # Extract the patch
    patch = extract_patch_from_wsi(wsi_path, position, PATCH_SIZE)

    if patch:
        # Save the patch as an image file
        patch_filename = f"{wsi_id}_{position[0]}_{position[1]}.png"
        patch_path = os.path.join(OUTPUT_DIR, patch_filename)
        patch.save(patch_path)

        # Save the caption in a text file
        caption_filename = f"{wsi_id}_{position[0]}_{position[1]}.txt"
        caption_path = os.path.join(OUTPUT_DIR, caption_filename)
        with open(caption_path, 'w') as caption_file:
            caption_file.write(caption)

        print(f"Extracted and saved patch and caption for {wsi_id} at position {position}")
    else:
        print(f"Failed to extract patch for {wsi_id} at position {position}")
```

This step creates the final PathGen-1.6M image-caption pairs.



## Usage of Trained PathGen-CLIP model

The trained PathGen-CLIP can be downloaded via this [**link**](https://pub-7a38cc906afa44a4a01533c288d0b1af.r2.dev/pathgenclip.pt).

```
pip install open_clip_torch
```

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='path/pathgen-clip.pt')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("example.png")).unsqueeze(0)
text = tokenizer(["An H&E image of tumor patch", "An H&E image of normal patch"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```
