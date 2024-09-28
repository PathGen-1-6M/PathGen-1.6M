---
license: cc-by-2.0
---
## Download the model

The trained PathGen-CLIP can be downloaded via this [**link**](https://pub-7a38cc906afa44a4a01533c288d0b1af.r2.dev/pathgenclip.pt).

## Usage of PathGen-CLIP

```bash
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
