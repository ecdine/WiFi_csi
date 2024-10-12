import torch
import torch.nn.functional as F

class KeypointLoss:
    def __init__(self, weight_ce=1, weight_l2=1, alpha=0.75, gamma=2.0):
        self.weight_ce = weight_ce
        self.weight_l2 = weight_l2
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs.transpose(1, 2), targets, reduction='none')  
        pt = torch.exp(-ce_loss)  
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  
        return focal_loss.mean()  

    def l2_loss(self, kpt_preds, kpt_gts):
        kpt_preds = kpt_preds.float()
        kpt_gts = kpt_gts.float()
        return F.mse_loss(kpt_preds, kpt_gts, reduction='mean')
    
    def compute_loss(self, src_logits, target_classes, pred, tgt_keypoints, tgt_area):
        loss_ce = self.focal_loss(src_logits, target_classes) 
        loss_l2 = self.l2_loss(pred, tgt_keypoints)
        total_loss = (loss_ce * self.weight_ce) +  (loss_l2 * self.weight_l2)
        print(f"cls_loss:{loss_ce.item()}"+f"L2_loss:{loss_l2.item()}")
        return total_loss