import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1Architecture
from Data import CarDataset,Compose
from Loss import YoloLoss
from utils import (
    NonMaximumSupression,
    mean_avg_precision,
    IOU,
    cellboxes_to_boxes,
    plot_image,
)

C =2 
S=7
B=2
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0.0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "weights/weight_yolov1.pth.tar"
# LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "Dataset/images"
LABEL_DIR = "Dataset/label"

seed = 123
torch.manual_seed(seed)

def get_nms_boxes(predict_grid_cell,
                  actual_grid_cell,
                  iou_threshold = 0.5,
                  confidence = 0.5,
                  type_box = "xywh",
                  start_idx=0):
    # coonverting to boxes
    # just only for one batch


    predict_boxes = cellboxes_to_boxes(predict_grid_cell,S=S,n_class=C)
    actual_boxes = cellboxes_to_boxes(actual_grid_cell,S=S,n_class=C)
    sample = predict_grid_cell.shape[0]
    total_predict_boxes = []
    total_actual_boxes = []
    
    for s in range(sample):
        predict_nms_boxes = NonMaximumSupression(predict_boxes[s],confidence=confidence,iou_threshold=iou_threshold,typebox=type_box)
        for nms_box in predict_nms_boxes:
            total_predict_boxes.append([start_idx] + nms_box)
        for box in actual_boxes[s]:
            if box[1]>confidence:
                total_actual_boxes.append([start_idx]+box)
        start_idx+=1
    return total_predict_boxes,total_actual_boxes,start_idx


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



def train(model : torch.nn.Module,
        loss_fn: torch.nn.Module,
        optim: torch.optim.Optimizer,
        train_iterator: DataLoader,
        valid_iterator: DataLoader,
        iou_threshold=0.5,
        confidence = 0.5,
        num_class=2,
        save_path = "weights/weight_yolov1.pth.tar",
        typebox="xywh",
        Device= "cpu",
        epochs=10):

    model.to(Device)
    start_idx_training = 0
    start_idx_val =  0 
    map_training_init,map_val_init,loss_training_init = 0,0,999999999
    for epoch in range(1, epochs + 1):
        pbar = tqdm(total=len(train_iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)

        training_loss = []
        # training_loss2 = []
        loss_val = 0  
        # set training mode
        model.train()
        total_predict_training_boxes,total_actual_training_boxes=[],[]
        total_predict_val_boxes,total_actual_val_boxes=[],[]

        # Loop through the training batch
        for batch, (image,label) in enumerate(train_iterator):
            image = image.to(Device)
            label = label.to(Device)
            predict_value = model(image)
            total_predict_training_batch_boxes,total_actual_training_batch_boxes,start_idx_training = get_nms_boxes(predict_value,label,
                                                                             iou_threshold,confidence,
                                                                             typebox,start_idx_training)
            total_predict_training_boxes.extend(total_predict_training_batch_boxes)
            total_actual_training_boxes.extend(total_actual_training_batch_boxes)
            # computing loss
            loss = loss_fn(predict_value,label)
            # loss2 = loss_fn1(predict_value,label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            training_loss.append(loss.item())
            # training_loss2.append(loss2.item())
            pbar.set_postfix(
                epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}", refresh=True)
            pbar.update()
        # ------ inference time --------
        y_predict_inference = None
        with torch.inference_mode():
            # Set the model to eval
            model.eval()
            # Loop through the validation batch
            for batch,(image,label) in enumerate(valid_iterator):
                image = image.to(Device)
                label = label.to(Device)
                y_predict_inference = model(image)
                loss_val += loss_fn(y_predict_inference,label).item()
                total_predict_val_batch_boxes,total_actual_val_batch_boxes,start_idx_val = get_nms_boxes(predict_value,label,iou_threshold,confidence,typebox,start_idx_val)
                total_predict_val_boxes.extend(total_predict_val_batch_boxes)
                total_actual_val_boxes.extend(total_actual_val_batch_boxes)
                # implement get box

        map_training = mean_avg_precision(total_predict_training_boxes,total_actual_training_boxes,iou_threshold,typebox,num_class).item()
        map_val = mean_avg_precision(total_predict_val_boxes,total_actual_val_boxes,iou_threshold,typebox,num_class).item()
        
        avg_loss = sum(training_loss)/len(training_loss)
        pbar.set_postfix(
            epoch=f" {epoch} training_loss = {round(avg_loss,4)}, map_training = {round(map_training,4)}, map_training_init = {map_training_init},  val_loss = {round(loss_val/len(valid_iterator),4)}, map_val = {round(map_val,4)}, map_val_ini = {map_val_init} ",refresh=False)
        pbar.close()
        if (map_training_init<=map_training and map_val_init<map_val)or(map_training_init<=map_training and loss_training_init>avg_loss):
            map_training_init = map_training
            if map_val_init<map_val:
                map_val_init=map_val
            if loss_training_init<avg_loss:
                loss_training_init=avg_loss
            check_point = {
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
                "map_train":map_training_init,
                "map_val":map_val_init
            }
            save_checkpoint(check_point,save_path)

    return model





transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])





def main():
    model = Yolov1Architecture(p=0.1,num_split=S,num_classes=C,num_B=B).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)


    loss_fn = YoloLoss(S=S,C=C,B=B)
    train_dataset = CarDataset(
        "Dataset/training_data.xlsx",
        transform=transform,
        img_path= IMG_DIR,
        label_path=LABEL_DIR,
        S=S,
        B=B,
        C=C
    )
    test_dataset = CarDataset(
        "Dataset/test_data.xlsx", transform=transform, img_path=IMG_DIR, label_path=LABEL_DIR,S=S,B=B,C=C
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )



    train(model,loss_fn,optimizer,train_loader,test_loader,num_class=C,Device=DEVICE,epochs=EPOCHS)

    # data = iter(train_loader)
    # image0,label0 = next(data)
    # print(image0.shape)
    # imag0 = image0.to(DEVICE)
    # label0 = label0.to(DEVICE)
    # predict = model(imag0)
    # total_predict_boxes,total_actual_boxes = get_nms_boxes(predict,label0)
    # print(mean_average_precision(total_predict_boxes,total_actual_boxes))
    # print(mean_avg_precision(total_predict_boxes,total_actual_boxes))


    return


if __name__ =="__main__":
    main()