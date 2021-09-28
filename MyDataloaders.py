import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SegDataset(Dataset):
    def __init__(self, parentDir, childDir , imageDir, maskDir, targetSize,load_to_RAM= False):
        self.imageList = glob.glob(parentDir + '/data_'+childDir+'/' + imageDir + '_'+childDir+'/*')
        self.imageList.sort()
        self.maskList = glob.glob(parentDir + '/data_'+childDir+'/' + maskDir + '_'+childDir+'/*')
        self.maskList.sort()
        self.targetSize = targetSize
        self.tensor_images = []
        self.tensor_masks = []
        self.load_to_RAM = load_to_RAM
        if self.load_to_RAM:# load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            self.tensor_masks = [self.get_tensor_mask(mask_path) for mask_path in self.maskList]
            print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:#if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images[index]
            y = self.tensor_masks[index]
        else:
            x=self.get_tensor_image(self.imageList[index])
            y=self.get_tensor_mask(self.maskList[index])
        return x, y

    def __len__(self):
        return len(self.imageList)

    def get_tensor_image(self, image_path):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize, 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        X = Image.open(image_path).convert('RGB')
        X = preprocess(X)
        return X
    def get_tensor_mask(self, mask_path):
        trfresize = transforms.Resize(self.targetSize, 2)
        trftensor = transforms.ToTensor()
        yimg = Image.open(mask_path).convert('L')
        y1 = trftensor(trfresize(yimg))
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)
        # y.squeeze_()
        return y

#TTR is Train Test Ratio
def trainTestSplit(dataset, TTR):
    '''This function split train test randomely'''
    print("dataset is splitted randomely")
    dataset_size = len(dataset)
    dataset_permutation = np.random.permutation(dataset_size)
    # print(dataset_permutation[:10])
    # trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    # valDataset = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    #
    trainDataset = torch.utils.data.Subset(dataset, dataset_permutation[:int(TTR * len(dataset))] )
    valDataset = torch.utils.data.Subset(dataset,dataset_permutation[int(TTR * len(dataset)):] )
    print("training indices first samples{}\n val indices first samples{}".format(trainDataset.indices[:5],valDataset.indices[:5]))
    # print(trainDataset.dataset[0])
    # exit(0)
    return trainDataset, valDataset