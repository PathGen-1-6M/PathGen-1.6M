# Representative Patch Extraction

## Step 1: Revise `svs_path_list.txt`
Update `svs_path_list.txt` to include the paths to your own whole slide images.

## Step 2: Use Cleaned WSI Report
Download and use the cleaned WSI report file: [wsi_caption_dict.json](https://github.com/PathGen-1-6M/PathGen-1.6M/blob/main/Patch_selection/wsi_caption_dict.json).

## Step 3: Install OpenSlide
To install OpenSlide, run the following command:

```bash
pip install openslide-python
```

## Step 4:  Start Extracting Representative Patches

```
python  wsi_representative_patch_extraction.py
```

