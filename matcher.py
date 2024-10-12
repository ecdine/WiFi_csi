# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 100, cost_oks: float = 10, cost_kpt: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_oks = cost_oks
        self.cost_kpt = cost_kpt
        

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1).float()
        #probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        #probas = probas.reshape(100)
        #print(max(probas))
        out_keypoint = outputs["pred_keypoints"].flatten(0, 1).float()
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_areas = torch.cat([v["areas"] for v in targets])
        tgt_keypoint = torch.cat([v["keypoints"] for v in targets]).float()
        tgt_keypoints_reshape = tgt_keypoint.reshape(tgt_keypoint.shape[0], -1, 3)  # [bs, 14, 3]
        kpt_pred_tmp = out_keypoint.clone().detach().reshape(out_keypoint.shape[0], -1, 3)#[bs, 14, 3]
        ##normalized_gt_keypoints = tgt_keypoints_reshape[..., :2]  # Removed normalization factor
        neg_cost = -(1 - out_prob + 1e-12).log() * (
            1 - 0.25) * out_prob.pow(2)
        pos_cost = -(out_prob + 1e-12).log() * 0.25 * (
            1 - out_prob).pow(2)
        cls_cost = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]
        cost_kpt = self.kpt_cost(kpt_pred_tmp, tgt_keypoints_reshape)
        kpt_pred_tmp = out_keypoint.clone().detach().reshape(out_keypoint.shape[0], -1, 3)
        cost_oks = self.oks_cost(kpt_pred_tmp, tgt_keypoints_reshape, tgt_areas) 

        C = self.cost_kpt * cost_kpt + self.cost_class * cls_cost + self.cost_oks * cost_oks
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["keypoints"]) for v in targets]
        ##sizes = len(targets["keypoints"])
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


    def kpt_cost(self, kpt_pred, gt_keypoints):
        kpt_cost = []
        for i in range(len(gt_keypoints)):
            kpt_pred_tmp = kpt_pred.clone()
            valid_kpt_flag = torch.ones(len(gt_keypoints),14)
            valid_flag = valid_kpt_flag[i] > 0
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(
                -1).expand_as(kpt_pred_tmp)
            kpt_pred_tmp[~valid_flag_expand] = 0
            cost = torch.cdist(
                kpt_pred_tmp.reshape(kpt_pred_tmp.shape[0], -1),
                gt_keypoints[i].reshape(-1).unsqueeze(0),
                p=1)
            avg_factor = torch.clamp(valid_flag.float().sum() * 2, 1.0)
            cost = cost / avg_factor
            kpt_cost.append(cost)
        kpt_cost = torch.cat(kpt_cost, dim=1)
        return kpt_cost
    
    def oks_cost(self, kpt_pred, gt_keypoints,gt_areas):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].
            gt_areas (Tensor): Ground truth mask areas. Shape [num_gt,].

        Returns:
            torch.Tensor: oks_cost value with weight.
        """
        self.sigmas = np.array([
                .26,
                .35,
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89], dtype=np.float32) / 10.0
        sigmas = torch.from_numpy(self.sigmas).to(kpt_pred.device)
        variances = (sigmas * 2)**2

        oks_cost = []
        for i in range(len(gt_keypoints)):
            squared_distance = \
                (kpt_pred[:, :, 0] - gt_keypoints[i, :, 0].unsqueeze(0)) ** 2 + \
                (kpt_pred[:, :, 1] - gt_keypoints[i, :, 1].unsqueeze(0)) ** 2
            valid_kpt_flag = torch.ones(len(gt_keypoints),14)
            vis_flag = (valid_kpt_flag[i] > 0).int()
            vis_ind = vis_flag.nonzero(as_tuple=False)[:, 0]
            num_vis_kpt = vis_ind.shape[0]
            assert num_vis_kpt > 0
            area = gt_areas[i]
            squared_distance0 = squared_distance / (area * variances * 2)
            squared_distance0 = squared_distance0[:, vis_ind]
            squared_distance1 = torch.exp(-squared_distance0).sum(
                dim=1, keepdim=True)
            oks = squared_distance1 / num_vis_kpt
            oks_cost.append(-oks)
        oks_cost = torch.cat(oks_cost, dim=1)
        return oks_cost
class Match():
    def __init__(self):
        pass
    def keypoint_match(outputs, targets, indices):
        src_keypoints = outputs['pred_keypoints']
        src_logits = outputs['pred_logits']
        tgt_ids = ([v["labels"] for v in targets])
        target_keypoints = [v["keypoints"].float() for v in targets]
        tgt_areas = ([v["areas"] for v in targets])
        target_classes = torch.full(src_logits.shape[:2], 1,
            dtype=torch.int64, device=src_logits.device)
        pred = torch.empty(0)
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        pred = pred.to(device=device)
        tgt = torch.empty(0)
        tgt = tgt.to(device=device)
        tgt_area = torch.empty(0)
        tgt_area =tgt_area.to(device=device)
        for batch_idx, (pred_indices, target_indices) in enumerate(indices):
            selected_targets1 = tgt_ids[batch_idx][target_indices]
            target_classes[batch_idx][pred_indices] = selected_targets1
            selected_preds = src_keypoints[batch_idx][pred_indices]
            selected_targets = target_keypoints[batch_idx][target_indices]
            selected_area = tgt_areas[batch_idx][target_indices]
            pred = torch.cat([pred,selected_preds],dim=0)
            tgt = torch.cat([tgt,selected_targets],dim=0)
            tgt_area = torch.cat([tgt_area,selected_area],dim=0)
        return src_logits, target_classes, pred, tgt, tgt_area



