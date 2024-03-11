import torch.nn as nn
import torch
# from utils import (
#     NonMaximumSupression,
#     mean_avg_precision,
#     IOU,
#     cellboxes_to_boxes,
#     plot_image,
# )

# kernel_size,output_kernel,stride,padding
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNB(nn.Module):
    def __init__(self, in_channel_dims,out_channel_dims,**kwargs) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(out_channel_dims)
        self.conv = nn.Conv2d(in_channel_dims,out_channels=out_channel_dims,**kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self,x):
        return self.leakyrelu(self.norm(self.conv(x)))
    
class Yolov1Architecture(nn.Module):
    def __init__(self,inchanel=3,p=0,**kwargs):
        super().__init__()
        self.norm = nn.BatchNorm2d(inchanel)
        self.architecture = architecture_config
        self.inchanel = inchanel
        self.backbone = self._conv_layer()
        self.dropout = p
        self.head = self._head_fc(**kwargs)
    def forward(self,x):
        out = self.head(self.backbone(self.norm(x)))
        return out
    def _conv_layer(self):
        inchanel = self.inchanel
        layer = []
        for i in self.architecture:
            if isinstance(i,tuple):
                layer += [CNNB(in_channel_dims=inchanel,out_channel_dims=i[1],kernel_size = i[0],stride = i[2],padding = i[3])]
                inchanel = i[1]
            elif isinstance(i,str):
                layer += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
            elif isinstance(i,list):
                for _ in range(i[-1]):
                    layer += [CNNB(in_channel_dims=inchanel,out_channel_dims=i[0][1],kernel_size = i[0][0],stride = i[0][2],padding = i[0][3])]
                    inchanel = i[0][1]
                    layer += [CNNB(in_channel_dims=inchanel,out_channel_dims=i[1][1],kernel_size = i[1][0],stride = i[1][2],padding = i[1][3])]
                    inchanel = i[1][1]
        return nn.Sequential(*layer)
    def _head_fc(self,num_split=7,num_classes=2,num_B=2):
        S, B, C = num_split, num_B, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 256),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(256, S * S * (C + B * 5)),
        )

def main():
    image = torch.rand(1,3,448,448)
    model  = Yolov1Architecture(num_split=7,num_classes=4,num_B=2)
    predict = model(image)
    a = predict.reshape(1,7,7,-1)
    print(a.shape)


    return


if __name__ =="__main__":
    main()
