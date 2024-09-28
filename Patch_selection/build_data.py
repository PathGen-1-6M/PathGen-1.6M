from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ov_patch_dataset_1wsi(Dataset):
    def __init__(self, wsi_obj, patch_cor_list, downsample_rate, patch_size=(224, 224),
                 resize=True, return_ori=False):
        self.wsi_obj = wsi_obj
        self.total_num=0
        self.patch_cor_list = patch_cor_list
        self.downsample_rate = downsample_rate
        self.patch_size = patch_size
        self.return_ori = return_ori
        if resize:
            self.transform = transforms.Compose([
                transforms.Resize((resize,resize), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])

    def __getitem__(self, index):

        corr = self.patch_cor_list[index]
        if len(corr)==4:
            x1, y1, x1_thu, y1_thu = corr
        else:
            x1, y1 = corr
            x1_thu, y1_thu = int(x1/self.downsample_rate), int(y1/self.downsample_rate)
        if hasattr(self.wsi_obj,'read_region'):
            img = self.wsi_obj.read_region((x1,y1),0,(self.patch_size,self.patch_size))
            if type(img) is not Image.Image:
                img = Image.fromarray(img).convert("RGB")
            else:
                img = img.convert("RGB")
        else:
            img = Image.fromarray(self.wsi_obj.read((x1, y1), size=(self.patch_size,self.patch_size)))
        img = self.transform(img)
        if self.return_ori:
            return img, x1,y1
        else:
            return img, x1_thu, y1_thu
    def __len__(self):
        return len(self.patch_cor_list)

