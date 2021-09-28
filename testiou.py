from Step1.Metrics import iou_my
import torch
outputs = torch.ones((2,1,3,3)).to('cpu')
labels = torch.ones((2,1,3,3)).to('cpu')
labels[0,0,0,0]=0

print(iou_my(outputs,labels))