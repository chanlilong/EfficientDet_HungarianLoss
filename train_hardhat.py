import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import glob
from model import Efficient_Detv2_S,Efficient_Detv2_S_conv
from dataset import hardhat_dataset,hardhat_collatefn,box_cxcywh_to_xyxy_np
from matcher import Criterion
from tensorboardX import SummaryWriter
from tqdm import tqdm

def inference():
    with torch.no_grad():
        img = data["image"][-1:]
        output = detr(img)
    
    clss_filter = output["pred_logits"].sigmoid().max(-1)[0]>0.35
    bboxes_pred = output["pred_boxes"][clss_filter].cpu().numpy()

    if len(bboxes_pred)>0:
        fig,ax = plt.subplots(1,1,figsize=(10,10))

        ax.imshow(img[0].permute(1,2,0).cpu().numpy())

        for box in bboxes_pred:
            box = np.clip(box_cxcywh_to_xyxy_np(box).reshape(-1,4),0,1)
            box*=np.array([[512,512,512,512]])
            xmin,ymin,xmax,ymax = box.flatten()
            rect = plt.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,color=(0,1,0),linewidth=3)
            ax.add_patch(rect)

        return fig
    else:
        return None

def write_to_tensorboard(itr,metrics,writer,log_type="train_losses/",detach=True):

    
    for key,value in metrics.items():
        name = log_type + key
        if detach:
            writer.add_scalar(name, value.detach().cpu().mean().numpy(), itr)
        else:
            writer.add_scalar(name, value.cpu().numpy(), itr)
            
            
writer = SummaryWriter('./tensorboard_logs/hardhat_effdetv2_reduced_3scales')
dataset = hardhat_dataset("/home/conda/RAID_5_14TB/DATASETS/hardhat")
dataloader = torch.utils.data.DataLoader(dataset,batch_size=16,collate_fn=hardhat_collatefn,shuffle=True,num_workers=4,pin_memory=True)

anchor_dict = np.load("cluster_hardhat_3scales_1anchor.npy",allow_pickle=True).item()
# detr = Efficient_Detv2_S(anchor_dict)
detr = Efficient_Detv2_S(anchor_dict)
detr.cuda()

criterion = Criterion(3)

# param_dicts = [
#     {
#         "params":
#             [p for n, p in detr.named_parameters() if "efficientnetv2_s_backbone" in n and p.requires_grad],
#         "lr": 2e-05,
#     },
#     {
#         "params": [p for n, p in detr.named_parameters() if "efficientnetv2_s_backbone" not in n and p.requires_grad],
#         "lr": 2e-04,
#     },

# ]

optimizer = torch.optim.AdamW(detr.parameters(),lr=1e-04, weight_decay=1e-04)


itr = 0
for e in tqdm(range(100),position=0, leave=True):
    for data in dataloader:
        
        optimizer.zero_grad()
        data = {k:([x.cuda() for x in v] if type(v)==list else v.cuda()) for k,v in data.items()}
        
        # with torch.cuda.amp.autocast():
        outputs = detr(data["image"])

        loss_dict = criterion(outputs,data)
        loss = sum(loss_dict.values())

        loss.backward()
        optimizer.step()

        itr+=1

        if itr%5==0:
            write_to_tensorboard(itr,loss_dict,writer)
            with torch.no_grad():
                max_probs = outputs["pred_logits"].sigmoid().max()
                writer.add_scalar("train_losses/max_prob", max_probs.detach().cpu().numpy(), itr)

        if itr%100==0:
            fig = inference()
            if fig is not None:
                writer.add_figure("images/predicted",fig,itr)

        if itr%1000==0 and itr!=1000:
            model_dict = {"params":detr.state_dict(),"optimizer":optimizer.state_dict(),"itr":itr}
            torch.save(model_dict,"./model_hardhat_3scales.pth")

model_dict = {"params":detr.state_dict(),"optimizer":optimizer.state_dict(),"itr":itr}
torch.save(model_dict,"./model_hardhat_3scales.pth")        