import torch
import torch.nn as nn
import torchvision

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.conv1 = nn.Conv2d(1,10,1,padding=0)
        self.rnn = nn.GRU(input_size=10,hidden_size=10,num_layers =1,batch_first=True)

    def forward(self,x):
        x = self.conv1(x)
        shape = x.shape
        print("original input",x.shape)
        # x_transposed = x.transpose(2,3)
        # print("rotated",x_transposed)

        x_viewed = x.view(shape[0],-1,shape[1]) #batch,sequence(H*W),inputsize(channels)
        # x_transposed = x_transposed.reshape(shape[0],-1,shape[1])
        print("flatten",x_viewed.shape)
        # print("flatten after rotated",x_transposed)
        x_viewed,h = self.rnn(x_viewed)
        # x_transposed,h = self.rnn(x_transposed)

        x_viewed = x_viewed.view(shape[0],-1,shape[2],shape[3])
        # x_transposed = x_transposed.view(*shape)
        # x = torch.cat((x,x_viewed,x_transposed),1)
        # print("viewd RNN output",x_viewed)
        # print("transposed RNN output",x_transposed)
        print("output shape",x_viewed.shape)

        return x_viewed #+x_transposed

model_ft = RNN()
input = torch.Tensor([[[[1,2],[3,4]]]])
# print()
output = model_ft(input)
# output = output.sum()
# print(output)
# output.backward
# loss = output+0.0
# print(model_ft.conv1.weight.grad)
# loss.backward()
# print(model_ft.conv1.weight.grad)
# write_to_left = input.view(1,1,-1).squeeze()
# up_to_down = input.transpose(2,3).reshape(1,1,-1).squeeze()
# print("write_to_left",write_to_left)
# print("up_to_down",up_to_down)
# input = torch.rot90(input,2,[2,3])
# write_to_left = input.reshape(1,1,-1).squeeze()
# up_to_down = input.transpose(2,3).reshape(1,1,-1).squeeze()
# print("write_to_left",write_to_left)
# print("up_to_down",up_to_down)

