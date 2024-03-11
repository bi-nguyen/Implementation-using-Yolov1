import torch
from utils import IOU

class YoloLoss(torch.nn.Module):
    def __init__(self,S=7,C=2,B=2,lambda_noobj=0.5,lambda_coord=5,type_box = "xywh") -> None:
        super().__init__()
        self.S = S
        self.C = C
        self.B = B
        self.mse = torch.nn.MSELoss(reduction="sum")
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.type_box = type_box
    def forward(self,predict_value,actual_value):
        '''
        predict_value : torch.tensor (batch_size,hidden_size)
        actual_value  : torch.tensor (batch_size,S,S,C+B*5)
        '''
        # convert predict_value 
        batch_size = predict_value.shape[0]
        predict_value = predict_value.reshape(batch_size,self.S,self.S,self.C+self.B*5)
        # extracting box1 and box2 from predict_value
        coord_box1 = predict_value[...,self.C+1:self.C+5]
        coord_box2 = predict_value[...,self.C+6:]
        coord_actual_box = actual_value[...,self.C+1:self.C+5]
        iou_box1 = IOU(coord_box1,coord_actual_box,self.type_box) # -> (batch_size,S,S,1)
        iou_box2 = IOU(predict_value[...,self.C+6:],coord_actual_box,self.type_box) # -> (batch_size,S,s,1)
        iou_stack = torch.cat((iou_box1.unsqueeze(0),iou_box2.unsqueeze(0)),dim=0)
        _,max_iou_box = torch.max(iou_stack,dim=0)
        mask = actual_value[...,self.C].unsqueeze(3)
        
        # computing coordinate loss (N,S,S,4) - > (N*S*S,4)
            # convert value w,h to square root
        final_coordinate_box = mask*(
                        (1-max_iou_box)*coord_box1 + max_iou_box*predict_value[...,self.C+6:]
        )
        coord_actual_box = mask*coord_actual_box
        final_coordinate_box[...,2:4] = torch.sign(final_coordinate_box[...,2:4])*torch.sqrt(
                                        torch.abs(final_coordinate_box[...,2:4]+1e-6))
        
        coord_actual_box[...,2:4]  = torch.sqrt(coord_actual_box[...,2:4])
        
        loss_coordinate = self.mse(torch.flatten(coord_actual_box,end_dim=-1),
                                   torch.flatten(final_coordinate_box,end_dim=-1))
        
        # computing confidence loss for grid cell containing object (N,S,S,1) -> (N*S*S*1)
        confidence_box1 = predict_value[...,self.C:self.C+1]
        confidence_box2 = predict_value[...,self.C+5:self.C+6]
        confidence_actual_box = actual_value[...,self.C:self.C+1]*mask
        final_confidence_box = mask*(
                        (1-max_iou_box)*confidence_box1+max_iou_box*confidence_box2
        )
        loss_confidence = self.mse(torch.flatten(confidence_actual_box),torch.flatten(final_confidence_box))


        # computing confidence loss for grid cell without object
        confidence_box1 = predict_value[...,self.C:self.C+1]
        confidence_box2 = predict_value[...,self.C+5:self.C+6]
        confidence_actual_box = actual_value[...,self.C:self.C+1]

        loss_confidence_noobj_1 = self.mse(torch.flatten((1-mask)*confidence_actual_box),torch.flatten((1-mask)*confidence_box1))
        loss_confidence_noobj_2 = self.mse(torch.flatten((1-mask)*confidence_actual_box),torch.flatten((1-mask)*confidence_box2))
        loss_confidence_noobj = loss_confidence_noobj_1+loss_confidence_noobj_2

        # Loss object
        class_predict_box = predict_value[...,:self.C] 
        class_actual_box = actual_value[...,:self.C]

        loss_class = self.mse(
            torch.flatten(class_actual_box*mask,end_dim=-1),
            torch.flatten(class_predict_box*mask,end_dim=-1)
            
        )

        total_loss = self.lambda_coord*loss_coordinate + loss_confidence + \
                     + self.lambda_noobj*(loss_confidence_noobj) + loss_class
        return total_loss



        



        
         