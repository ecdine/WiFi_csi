import os
import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer import Transformer, MLP
from position_encoding import PositionEncoding
from wipose import WifiPoseDataset
from loss import KeypointLoss
from matcher import HungarianMatcher, Match
from torch.utils.tensorboard import SummaryWriter 

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.batch_norm = nn.BatchNorm2d(60)
        self.pos_embed = PositionEncoding(temperature=10000)
        self.query_embed = nn.Embedding(50, 60)
        self.transformer = Transformer()
        self.linear_class = nn.Linear(in_features=60, out_features=2)
        self.MLP = MLP(60, 60, 42, 3)
        self.matcher = HungarianMatcher(cost_class=20.0, cost_oks=0.001, cost_kpt=70.0)

    def forward(self, img):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        mask = torch.ones(img.size(0), 180).to(device)
        img = self.batch_norm(img)
        pos = self.pos_embed.get_encoding(seq_len=29, d=60).to(device)
        query = self.query_embed.weight.to(device)
        output, memory = self.transformer(img, mask, query, pos)
        output = output.squeeze(0)
        outputs = {"pred_logits": self.linear_class(output), "pred_keypoints": self.MLP(output)}
        return outputs

if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('runs/TransformerModelTraining')  
    batchsize = 64
    epochs = 300
    dataset = WifiPoseDataset(dataset_root='/megadisk/fanghengyu/pert/data/train_data', mode='train')
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=WifiPoseDataset.custom_collate_fn, num_workers=8)
    model = TransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    loss_calculator = KeypointLoss(weight_ce=6.0, weight_l2=10.0, alpha=0.75, gamma=2.0)
    checkpoint_path = '/megadisk/fanghengyu/3D-Pose/path/train_50_200.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print("成功加载模型")
    else:
        print("开始新的训练")
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        for i, (img, targets) in enumerate(pbar):
            img = img.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            outputs = model(img)
            with torch.no_grad():
                indices = model.matcher(outputs, targets)
            src_logits, target_classes, pred, tgt_keypoints, tgt_area = Match.keypoint_match(outputs, targets, indices)
            loss = loss_calculator.compute_loss(src_logits, target_classes, pred, tgt_keypoints, tgt_area)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)  # 记录损失
            pbar.set_description(f'Epoch {epoch+1} Loss: {loss.item():.4f}')
        if (epoch + 1) % 10 == 0:
            Pth = f'path/train_50_{epoch + 1}.pth'
            torch.save(model.state_dict(), Pth)
    writer.close()  # 关闭 TensorBoard
