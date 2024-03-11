import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
class CarDataset(Dataset):
    def __init__(self,file_path,img_path,label_path,transform=None,S=7,C=2,B=2):
        super().__init__()
        self.file_path = pd.read_excel(file_path)
        self.img_path  = img_path
        self.label_path = label_path
        self.S = S
        self.C = C
        self.B = B
        self.transform = transform
    def __len__(self):
        return self.file_path.shape[0]
    def __getitem__(self, index):
        img = Image.open(self.img_path+"/"+self.file_path["image"].iloc[index])
        label_path = self.label_path+"/"+self.file_path["label"].iloc[index]
        boxes=[]
        with open(label_path,"r") as f:
            for box in f.readlines():
                box = box.replace("\n","")
                class_label,x,y,w,h = [float(b) if float(b)!=int(float(b)) else int(b)
                            for b in box.split()]
                
                boxes.append([class_label,x,y,w,h])
        boxes = torch.tensor(boxes)
        if self.transform:
            img,boxes = self.transform(img,boxes)       
        matrix = torch.zeros((self.S,self.S,self.C+self.B*5))
        for box in boxes:
            class_label,x,y,w,h = box.tolist()
            class_label = int(class_label)
            i,j = int(self.S*y),int(self.S*x)
            x_cell,y_cell = self.S*x-j,self.S*y-i
            h_cell,w_cell = h*self.S,w*self.S
            if matrix[i,j,self.C]==0:
                matrix[i,j,self.C] =1
                matrix[i,j,class_label]=1
                matrix[i,j,self.C+1:self.C+5] = torch.tensor([x_cell,y_cell,w_cell,h_cell])
        return img,matrix

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

def main():
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    file_path = "Dataset\VehicleDataset.xlsx"
    data = CarDataset(file_path=file_path,img_path="Dataset\image",label_path="Dataset\label",transform=transform)
    train_loader = DataLoader(dataset=data,batch_size=3)
    sample = iter(train_loader)
    total_image,total_label = next(sample)
    total_label_new = total_label[0].reshape(1,49,-1)

    # torch.save(total_label,"UsingTestCase1.pt")
    # torch.save(total_image,"ImageTestCase1.pt")
    # for i in range(total_label_new.shape[0]):
    #     for grid in range(49):
    #         if total_label_new[i,grid,4]==1.0:
    #             print(total_label_new[i,grid,:])

    return



if __name__=="__main__":
    main()
    