import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import glob
import PIL.Image as Image

def box_cxcywh_to_xyxy_np(x):
    x_c, y_c, w, h = x[...,0:1],x[...,1:2],x[...,2:3],x[...,3:]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
    return np.concatenate(b,axis=-1)

def box_xyxy_to_cxcywh_np(x):
    x0, y0, x1, y1 = x[...,0:1],x[...,1:2],x[...,2:3],x[...,3:]
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return np.concatenate(b,axis=-1)

class hardhat_dataset(torch.utils.data.Dataset):
    
    def __init__(self,root = "/home/lilong/HDD_3tb/DATASETS_OUTSIDE/DATASETS/hardhat",datatype="train"):
        super().__init__()
        

        self.labels = pd.read_csv(root+f"/{datatype}/_annotations.csv")
        self.labels[["xmin","ymin","xmax","ymax"]] = self.labels[["xmin","ymin","xmax","ymax"]]/self.labels[["width","height","width","height"]].values
        self.filenames = self.labels["filename"]
        self.file_directories = [f"{root}/{datatype}/{x}" for x in self.filenames]
        self.class_uniques = self.labels["class"].unique()
        self.class2int = {v:k for k,v in enumerate(self.class_uniques)}
        self.int2class = {v:k for k,v in self.class2int.items()}

        self.labels["class"] = self.labels["class"].apply(lambda x: self.class2int[x])
        
        
    def __len__(self):
        return len(self.file_directories)
    
    def __getitem__(self,idx):
    
        file_idx = self.filenames[idx]
        label_idx = self.labels[self.labels["filename"]==file_idx]
        img = Image.open(self.file_directories[idx]).convert("RGB")
        img = np.asarray(img.resize((512,512)))
        
        
        data = {}
        data["image"] = torch.as_tensor(img).permute(2,0,1).unsqueeze(0)
        data["labels"] = label_idx["class"].values
        data["boxes"]  = np.clip(box_xyxy_to_cxcywh_np(label_idx[["xmin","ymin","xmax","ymax"]].values),0,1)
        
        return data
    
def hardhat_collatefn(batch):
    
    data = {}
    
    data["image"] = torch.cat([x["image"]/255.0 for x in batch],dim=0).to(torch.float32)
    data["labels"] = [torch.as_tensor(x["labels"],dtype=torch.int32) for x in batch]
    data["boxes"] = [torch.as_tensor(x["boxes"],dtype=torch.float32) for x in batch]
    
    return data
