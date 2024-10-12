from transformer import Transformer
from transformer import MLP
from position_encoding import PositionEncoding
from wipose import WifiPoseDataset
from torch import nn
from torch.utils.data import DataLoader
import torch.optim
import torch
from matcher import HungarianMatcher
import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.batch_norm = nn.BatchNorm2d(60)
        self.pos_embed = PositionEncoding(temperature = 10000)   
        self.query_embed = nn.Embedding(50, 60)
        self.transformer = Transformer() 
        self.linear_class = nn.Linear(in_features=60, out_features=2)
        self.MLP = MLP(60, 60, 42, 3)
        self.matcher = HungarianMatcher(cost_class=20.0, cost_oks=0.001, cost_kpt=70.0)

    def forward(self, img):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        mask = torch.ones(img.size(0), 180).to(device)
        img = self.batch_norm(img)
        pos = self.pos_embed.get_encoding(seq_len = 29, d = 60).to(device)
        query = self.query_embed.weight
        self.query_embed.weight = self.query_embed.weight.to(device)
        output, memory = self.transformer(img, mask, query, pos)
        output = output.squeeze(0)
        outputs = {}
        outputs["pred_logits"] = self.linear_class(output)
        outputs["pred_keypoints"] = self.MLP(output)
        return outputs
    
if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    batchsize = 1
    epochs = 1
    dataset = WifiPoseDataset(dataset_root='/megadisk/fanghengyu/pert/data/test_data', mode='test')
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn= WifiPoseDataset.custom_collate_fn, num_workers=6)
    model = TransformerModel().to(device)
    checkpoint_path = '/megadisk/fanghengyu/3D-Pose/path/train_50_300.pth'  
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print("成功加载模型")
    model.eval()
    i = 0
    j = 0
    accurate_keypoints = 0
    total_keypoints = 0
    all_distances = []

    with torch.no_grad():
        for img, targets in tqdm(dataloader, desc="Processing batches"):
            j += 1
            img = img.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(img)
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.55
            keypoints = outputs['pred_keypoints'][0, keep]
            keypoints = keypoints.reshape(-1, 14, 3)
            keypoints = keypoints.detach().cpu().numpy()
            tgt_keypoint = torch.cat([v["keypoints"] for v in targets]).float()
            tgt_keypoint = tgt_keypoint.reshape(-1, 14, 3).detach().cpu().numpy()
            if keypoints.shape[0] == tgt_keypoint.shape[0]:
                i += 1
            row_ind, col_ind = linear_sum_assignment(cdist(keypoints.reshape(-1, 42), tgt_keypoint.reshape(-1, 42), metric='euclidean'))
            keypoints = keypoints[row_ind]
            tgt_keypoint = tgt_keypoint[col_ind]
            distances = np.linalg.norm(keypoints - tgt_keypoint, axis=-1)
            if distances.size > 0:
                all_distances.append(distances)
            avg_distances = distances.mean(axis=1) if distances.size > 0 else []
            threshold = 0.2
            accurate_matches = distances < threshold
            accurate_keypoints += accurate_matches.sum()
            total_keypoints += keypoints.size / 3
    
    total_accuracy = accurate_keypoints / total_keypoints
    final_average_distance = np.mean([dist.mean() for dist in all_distances])
    
    print(f"整体关键点准确率: {total_accuracy * 100:.2f}%", f"召回率： {i / j * 100:.2f}%")
    print(f"平均关键点距离: {final_average_distance:.4f}")