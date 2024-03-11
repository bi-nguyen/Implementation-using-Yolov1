import torch 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def IOU(box1,box2,type_box = "xywh"):
    '''
    boxe1 (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
    boxe2(tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    type_box (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    return
        tensor: value
    '''
    if type_box == "xywh": # convert xywh to xyxy

        box1_x1 = box1[...,0:1]-box1[...,2:3]/2
        box1_y1 = box1[...,1:2]-box1[...,3:4]/2 
        box1_x2 = box1[...,0:1]+box1[...,2:3]/2
        box1_y2 = box1[...,1:2]+box1[...,3:4]/2 

        #-------------------------------------------
        box2_x1 = box2[...,0:1]-box2[...,2:3]/2
        box2_y1 = box2[...,1:2]-box2[...,3:4]/2 
        box2_x2 = box2[...,0:1]+box2[...,2:3]/2
        box2_y2 = box2[...,1:2]+box2[...,3:4]/2 
    elif type_box == "xyxy":
        box1_x1 = box1[...,0:1]
        box1_y1 = box1[...,1:2]
        box1_x2 = box1[...,2:3]
        box1_y2 = box1[...,3:4] 

        #-------------------------------------------
        box2_x1 = box2[...,0:1]
        box2_y1 = box2[...,1:2]
        box2_x2 = box2[...,2:3]
        box2_y2 = box2[...,3:4]    
    else:
        return "your type box is wrong"  

    # Find out coordinate of intersection
    x1 = torch.max(box1_x1,box2_x1)
    x2 = torch.min(box1_x2,box2_x2)
    y1 = torch.max(box1_y1,box2_y1)
    y2 = torch.min(box1_y2,box2_y2)
    # computing area of each box
    area_box1 = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    area_box2 = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))

    area_intersection = (x2-x1).clamp(0)*(y2-y1).clamp(0)
    return area_intersection/(area_box1+area_box2-area_intersection+1e-6)
    
# used for inference phase
def NonMaximumSupression(bbox,iou_threshold=0.5,confidence=0.5,typebox ="xywh"):
    '''
    input:
        bbox(list): containts list of bouding box [class_pred, prob_score, x1, y1, x2, y2] - the number of boxes in a batch size
        iou_threshold (float): threshold where predicted bboxes is correct
        confidence (float): threshold to remove predicted bboxes (independent of IoU) 
        typebox (str): "midpoint" or "corners" used to specify bboxes
    return:
        list bounding box
    '''
    # to get rid of bounding boxes with low confidence

    bbox = [box for box in bbox if box[1]>confidence]
    bbox.sort(key = lambda s: s[1],reverse=True)
    final_box = []
    while bbox:
        actual_box = bbox.pop(0)
        bbox = [box for box in bbox 
                if box[0] != actual_box[0] or
                IOU(torch.tensor(actual_box[2:]),torch.tensor(box[2:]),type_box=typebox)<iou_threshold
                ]
        final_box.append(actual_box)

    return final_box


def mean_avg_precision(predict_boxes,actual_boxes,iou_threshold=0.5,typebox="xywh",num_classes=2):
    '''
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    '''
    average_precisions = []
    epsilon = 1e-6


    for n in range(num_classes):
        predict_box = [box for box in predict_boxes if box[1]==n]
        actual_box = [box for box in actual_boxes if box[1]==n]
        amount_of_actual_box_per_image =  Counter([gt[0] for gt in actual_box])
        amount_of_actual_box_per_image_matrix = {key:torch.zeros(value) for key,value in amount_of_actual_box_per_image.items()}
        total_value = len(actual_box)
        TP = torch.zeros(len(predict_box))
        FP = torch.zeros(len(predict_box))
      # sorting bounding box follow confidence
        predict_box.sort(key= lambda s: s[2],reverse=True)
        for pred_idx,pred_box in enumerate(predict_box):
            actual_box_in_an_image = [box for box in actual_box if box[0]==pred_box[0]]
            iou_value = 0
            idx_value = 0
            for idx,actual in enumerate(actual_box_in_an_image):
                iou_result = IOU(torch.tensor(pred_box[3:]),torch.tensor(actual[3:]),type_box=typebox)
                if iou_value==None or iou_value<iou_result:

                    iou_value = iou_result
                    idx_value = idx

            if iou_value>iou_threshold:
                if amount_of_actual_box_per_image_matrix[pred_box[0]][idx_value]==0:
                    TP[pred_idx]=1
                    amount_of_actual_box_per_image_matrix[pred_box[0]][idx_value]=1
                else:
                    FP[pred_idx]=1
            else:
                FP[pred_idx]=1
        TP_cumsum = torch.cumsum(TP,dim=0)
        FP_cumsum = torch.cumsum(FP,dim=0)
        recall = TP_cumsum/(total_value+epsilon)
        precision = TP_cumsum/(TP_cumsum+FP_cumsum+epsilon)
        recall = torch.cat((torch.tensor([0]),recall))
        precision = torch.cat((torch.tensor([1]),precision))
        average_precisions.append(torch.trapz(precision, recall))
    return sum(average_precisions)/len(average_precisions)

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    color = ["r","b","g","yellow"]
    for box in boxes:
        c = box[0]
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor=color[int(c)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
        
def convert_cellboxes(predictions, S=7,n_class=2):
    '''
    predictions : (batchsizes,1470)
    
    '''
    predictions = predictions.to("cpu")
    batch_size= predictions.shape[0]
    predictions = predictions.reshape(batch_size,S,S,-1)
    s = predictions.shape
    # Looking for best box
    box1 = predictions[...,n_class+1:n_class+1+4]
    box2 = predictions[...,n_class+4+2:n_class+4+2+4]
    confidence_box = torch.cat((predictions[...,n_class].unsqueeze(0),predictions[...,n_class+5].unsqueeze(0)),dim=0)
    best_box = confidence_box.argmax(0).unsqueeze(-1)
    best_coordinate_box = box1*(1-best_box)+box2*best_box

    # convert cell size to normal size
    matrix_size = torch.arange(S).repeat(batch_size,S,1).unsqueeze(-1)
    x= (best_coordinate_box[...,:1]+matrix_size)/S
    y = (best_coordinate_box[...,1:2]+matrix_size.permute(0,2,1,3))/S
    w_h = best_coordinate_box[...,2:]/S
    coordinate = torch.cat((x,y,w_h),dim=-1)
    classes = predictions[...,:n_class].argmax(-1).unsqueeze(-1)
    confidences = torch.max(predictions[...,n_class],predictions[...,n_class+5]).unsqueeze(-1)
    return torch.cat((classes,confidences,coordinate),dim=-1) # Batch_size,7,7,1+1+4





def cellboxes_to_boxes(out, S=7,n_class=2):
    convert_cellboxes_result = convert_cellboxes(out,S,n_class=n_class).reshape(out.shape[0],S*S,-1)
    convert_cellboxes_result[...,0] = convert_cellboxes_result[...,0].long()
    total_boxes = []
    for batch in range(convert_cellboxes_result.shape[0]):
        boxes_per_batch = []
        for grid in range(S*S):
            boxes_per_batch.append([x.item() for x in convert_cellboxes_result[batch,grid,:]])
        total_boxes.append(boxes_per_batch)
    return total_boxes






def main():

    box1 = torch.tensor([0.8, 0.1, 0.2, 0.2])
    box2 = torch.tensor([0.9, 0.2, 0.2, 0.2])
    iou_result = IOU(box1,box2)

    # nms test
    t1_boxes = [
                [1, 1, 0.5, 0.45, 0.4, 0.5],
                [1, 0.8, 0.5, 0.5, 0.2, 0.4],
                [1, 0.7, 0.25, 0.35, 0.3, 0.1],
                [1, 0.05, 0.1, 0.1, 0.1, 0.1],
            ]
    t2_boxes = [
                [1, 0.6, 0.5, 0.45, 0.4, 0.5],
                [2, 0.9, 0.5, 0.5, 0.2, 0.4],
                [1, 0.8, 0.25, 0.35, 0.3, 0.1],
                [1, 0.05, 0.1, 0.1, 0.1, 0.1],
            ]
    bbox = NonMaximumSupression(t1_boxes)
    bboxes = NonMaximumSupression(
            t1_boxes,
            confidence=0.2,
            iou_threshold=7 / 20,
            typebox="xywh",
        )
    a = [1,2,3,4,5,1,3,5,7]
    a1=Counter(a)
    b = {key:torch.zeros(val) for key,val in a1.items()}
    c= torch.tensor([1,0,1,0,1,0,1,1,0])

    t4_preds = [
        [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]

    t4_targets = [
        [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    t3_correct_mAP = 0

    mean_avg_prec = mean_avg_precision(
        t4_preds,
        t4_targets,
        iou_threshold=0.5,
        typebox="xywh",
        num_classes=1,
    )
    print(5 / 18-mean_avg_prec)



if __name__ == "__main__":
    main()