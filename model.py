import timm
import torch

from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from torch.nn import functional as F


class resnet_backbone(torch.nn.Module):
    def __init__(self,single_scale=False,dims=256):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        
        self.activation = {}
#         self.layer_names = [f"layer{i+1}" for i in range(4)]
        self.single_scale = single_scale
        if single_scale:
            self.model.layer4.register_forward_hook(self.get_activation('layer4'))
            self.conv4 = torch.nn.Conv2d(2048,dims,1,1)
        else:
            self.model.layer4.register_forward_hook(self.get_activation('layer4'))
            self.model.layer3.register_forward_hook(self.get_activation('layer3'))
            self.model.layer2.register_forward_hook(self.get_activation('layer2'))
            self.model.layer1.register_forward_hook(self.get_activation('layer1'))
            
            self.conv4 = torch.nn.Conv2d(2048,dims,1,1)
            self.conv3 = torch.nn.Conv2d(1024,dims,1,1)
            self.conv2 = torch.nn.Conv2d(512,dims,1,1)
        
    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook
        
    def forward(self,x):
        _ = self.model.forward_features(x)
        
        if self.single_scale:
            l4 = self.activation["layer4"].to(x.device)
            self.activation = {}

            return self.conv4(l4)
        else:
            l1 = self.activation["layer1"].to(x.device) 
            l2 = self.activation["layer2"].to(x.device)
            l3 = self.activation["layer3"].to(x.device) 
            l4 = self.activation["layer4"].to(x.device)
        
            self.activation = {}

            return l1,F.interpolate(l1,scale_factor=0.75),self.conv2(l2),self.conv3(l3),self.conv4(l4)
        
class efficientnetv2_s_backbone(torch.nn.Module):
    def __init__(self,single_scale=False,dims=256):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True)
        
        self.activation = {}
#         self.layer_names = [f"layer{i+1}" for i in range(4)]
        self.single_scale = single_scale
        blocks = {n:m for n,m in self.model.blocks.named_children()} #0,1,2,3,4,5
        if single_scale:
            blocks["5"].register_forward_hook(self.get_activation('layer5'))
            self.conv5 = torch.nn.Conv2d(256,dims,1,1)
        else:

            blocks["5"].register_forward_hook(self.get_activation('layer5'))
            blocks["4"].register_forward_hook(self.get_activation('layer4'))
            blocks["3"].register_forward_hook(self.get_activation('layer3'))
            blocks["2"].register_forward_hook(self.get_activation('layer2'))
            blocks["1"].register_forward_hook(self.get_activation('layer1'))
            
            self.conv5 = torch.nn.Conv2d(256,dims,1,1)
            self.conv4 = torch.nn.Conv2d(160,dims,1,1)
            self.conv3 = torch.nn.Conv2d(128,dims,1,1)
            self.conv2 = torch.nn.Conv2d(64,dims,1,1)
            self.conv1 = torch.nn.Conv2d(48,dims,1,1)
        
    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook
        
    def forward(self,x):
        _ = self.model.forward_features(x)
        
        if self.single_scale:
            l5 = self.activation["layer5"].to(x.device)
            self.activation = {}

            return self.conv5(l4)
        else:
            l1 = self.activation["layer1"].to(x.device)
            l2 = self.activation["layer2"].to(x.device) 
            l3 = self.activation["layer3"].to(x.device)
            l4 = self.activation["layer4"].to(x.device) 
            l5 = self.activation["layer5"].to(x.device)
        
            self.activation = {}

            return self.conv1(l1),self.conv2(l2),self.conv3(l3),self.conv4(l4),self.conv5(l5)

from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
#                                padding, dilation, groups=in_channels, bias=False)
        
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
#                                    stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.dw_conv = DepthwiseConv2d(in_channels, out_channels,kernel_size,stride,padding,dilation)
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        # x = self.depthwise(inputs)
        # x = self.pointwise(x)
        
        x = self.dw_conv(inputs)
        x = self.bn(x)
        return self.act(x)
    
# class DepthwiseConvBlock(nn.Module):
#     """
#     Depthwise seperable convolution. 
    
    
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
#         super(DepthwiseConvBlock,self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
#                                padding, dilation, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
#                                    stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
#         self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
#         self.act = nn.LeakyReLU(inplace=True)
        
#     def forward(self, inputs):
#         x = self.depthwise(inputs)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.act(x)
        
        
class BiFPN(nn.Module):
    def __init__(self, in_channels ,out_channels):
        super(BiFPN, self).__init__()
        self.num_channels = out_channels

        self.conv7up = DepthwiseConvBlock(in_channels,out_channels)
        self.conv6up = DepthwiseConvBlock(in_channels,out_channels)
        self.conv5up = DepthwiseConvBlock(in_channels,out_channels)
        self.conv4up = DepthwiseConvBlock(in_channels,out_channels)
        self.conv3up = DepthwiseConvBlock(in_channels,out_channels)
        self.conv4dw = DepthwiseConvBlock(in_channels,out_channels)
        self.conv5dw = DepthwiseConvBlock(in_channels,out_channels)
        self.conv6dw = DepthwiseConvBlock(in_channels,out_channels)
        self.conv7dw = DepthwiseConvBlock(in_channels,out_channels)
        
    def forward(self, inputs):
        num_channels = self.num_channels
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs #imgsize: p3: big --> p7: small

        # upsample network
        P7_up = self.conv7up(P7_in)
        P6_up = self.conv6up(P6_in+F.interpolate(P7_up, P6_in.size()[2:],mode = "bilinear"))
        P5_up = self.conv5up(P5_in+F.interpolate(P6_up, P5_in.size()[2:],mode = "bilinear"))
        P4_up = self.conv4up(P4_in+F.interpolate(P5_up, P4_in.size()[2:],mode = "bilinear"))
        P3_out = self.conv3up(P3_in+F.interpolate(P4_up, P3_in.size()[2:],mode = "bilinear"))

        # fix to downsample by interpolation
        # downsample networks
        P4_out = self.conv4dw(P4_in + P4_up+F.interpolate(P3_out, P4_up.size()[2:],mode = "bilinear"))
        P5_out = self.conv5dw(P5_in + P5_up+F.interpolate(P4_out, P5_up.size()[2:],mode = "bilinear"))
        P6_out = self.conv6dw(P6_in + P6_up+F.interpolate(P5_out, P6_up.size()[2:],mode = "bilinear"))
        P7_out = self.conv7dw(P7_in + P7_up+F.interpolate(P6_out, P7_up.size()[2:],mode = "bilinear"))

        
        return P3_out, P4_out, P5_out,P6_out, P7_out

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes,n_anchors):
        super().__init__()
        self.pred = nn.Sequential(
                                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,bias=False),
                                    nn.BatchNorm2d(in_channels),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv2d(in_channels, (num_classes + 4) * n_anchors, kernel_size=1), #Batch,n_anchors,w,h,(prob,x,y,w,h,c1,c2,c3.....cn)
                                    )
        self.n_anchors = n_anchors
        self.num_classes = num_classes

    def forward(self, x):
        # return (self.pred(x).reshape(x.shape[0], self.n_anchors, self.num_classes + 4, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2))   
        return self.pred(x)

    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    
    
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1)-torch.log(x2)    

class Efficient_Detv2_S(nn.Module):
    
    def __init__(self,anchor_dictionary,n_classes=3):
        super().__init__()
        
        #clusters_dic = {"anchor_boxes":cluster_centers,"N_anchors":N_anchors,"N_scales":N_scales}

        self.setup_anchors(anchor_dictionary)
        # self.cnn_backbone = efficientnetv2_s_backbone()
        self.cnn_backbone = resnet_backbone()
        self.fpn = BiFPN(256,256)
        # self.SE_nets = nn.ModuleList([SELayer(64) for _ in range(self.n_scales)])
        # self.scale_pred = nn.ModuleList([ScalePrediction(256,n_classes,self.n_anchors) for _ in range(self.n_scales)])
        self.bbox_layer = nn.Sequential(nn.Linear(256,64),nn.LeakyReLU(inplace=True),nn.LayerNorm(64),nn.Linear(64,4))
        self.clss_layer = nn.Sequential(nn.Linear(256,64),nn.LeakyReLU(inplace=True),nn.LayerNorm(64),nn.Linear(64,n_classes))
        self.n_classes=n_classes
        
    def setup_anchors(self,anchor_dictionary):
        self.anchors = anchor_dictionary["anchor_boxes"]
        self.anchors.sort()
        self.anchors = self.anchors[::-1]
        self.n_anchors = anchor_dictionary["N_anchors"]
        self.n_scales = anchor_dictionary["N_scales"]
        self.anchors = self.anchors.reshape(self.n_scales,self.n_anchors,2)
        # print(self.anchors)
    def forward(self,x):
        
        scale_features = self.cnn_backbone(x)
        fpn_features = self.fpn(scale_features)
        fpn_features = [fpn_features[i] for i in [1,3,4]]
        # fpn_features = scale_features
        
        # fpn_features = [SE(fpn_features[i]) for i,SE in enumerate(self.SE_nets)]
            

        preds = []
        
        boxes = []
        logits = []
        for f,anchor_i in zip(fpn_features,self.anchors):
            # pred = layer(f)
            # print(f.shape)
            # torch.Size([1, 256, 128, 128])
            # torch.Size([1, 256, 64, 64])
            # torch.Size([1, 256, 32, 32])
            # torch.Size([1, 256, 32, 32])
            # torch.Size([1, 256, 16, 16])
            b,C,h,w = f.shape
            
            X,Y = torch.meshgrid(torch.linspace(0,1,w+1),torch.linspace(0,1,h+1))
            
            bbox = self.bbox_layer(f.permute(0,2,3,1).flatten(1,2)) #b,c,h,w ->b,h,w,c ->b,hw,c -> b,hw,4
            bbox_final = torch.zeros_like(bbox)
            anchor_i = torch.tensor(anchor_i,dtype=torch.float32).to(x.device)

            bbox_final[...,2:4] = (bbox[...,2:4] + inverse_sigmoid(anchor_i.expand_as(bbox[...,2:4]))).sigmoid()
            bbox_final[...,0:1] = bbox[...,0:1].sigmoid() + Y[:w,:w].flatten().unsqueeze(0).unsqueeze(2).to(x.device)
            bbox_final[...,1:2] = bbox[...,1:2].sigmoid() + X[:h,:h].flatten().unsqueeze(0).unsqueeze(2).to(x.device)         

            boxes.append(bbox_final)
            
            
            logit = self.clss_layer(f.permute(0,2,3,1).flatten(1,2))
            logits.append(logit)
            

        returns = {}
        returns["pred_logits"] = torch.cat(logits,1)
        returns["pred_boxes"] = torch.clamp(torch.cat(boxes,1),0,1)
        return returns
    
 