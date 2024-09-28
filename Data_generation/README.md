# PathGen-1.6M Data Processing

> ### Directly Obtain Data from Our PathGen-1.6M

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



# Pipelines for Generating Your Own Data

If you want to reuse our pipeline based on your own data, we provide a simple overall pipeline code for reference.

## Step 1: Representative Patch Extraction

Refer to the **Representative Patch Extraction** section to extract representative samples from your data.

## Step 2: Train the Caption Generation Model

- Download samples from PathCap, Quilt-1M, and OpenPath.
- Refer to the appropriate documentation (xxx) to prepare the data to match the input format required by LLaVA.

### Phase-1 Training

Run the following command:

```bash
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path path_to/pathgen_initial.json \
    --image_folder path_to/pathgen_initial/images \
    --vision_tower path_to/pathgen_clip \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/pathgen-llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

### Generate Detailed Captions Using GPT-4V

Follow the guidelines in our paper to generate detailed captions using GPT-4V. Below is the prompt for GPT-4V:

```
This is a microscopic image of cells or tissues. Please first describe the morphological characteristics of the cells or tissues in the image, and then supplement and correct it according to the additional description provided, but only include features that are observable in the image.

Be aware that some details in the additional description might not be present or identifiable in the image. Focus solely on the characteristics of the cells or tissues. Do not mention you have additional description such as 'considering the additional description', 'based on the given description'. Do not ask for other information or give unrelated response.

Additional description: {description}
```

Prepare the generated detailed captions in LLaVA format.

### Phase-2 Training

Run the following command to complete the training:

```bash
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path path_to/pathgen_initial.json \
    --image_folder path_to/pathgen_desp/images \
    --vision_tower path_to/pathgen_clip \
    --pretrain_mm_mlp_adapter ./checkpoints/pathgen-llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/pathgen-llava-v1.5-13b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

```





