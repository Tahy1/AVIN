import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18
from einops import rearrange
from models.audnet import ResNet22

class Audio_Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.fc2 = nn.Linear(512, 512, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class AVIN(nn.Module):
    def __init__(self, tp, tn):
        super().__init__()

        # Net components
        self.v_encoder = resnet18('visual', "pretrained/resnet18-f37072fd.pth")
        self.a_encoder = ResNet22()
        self.v_proj = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1 ,padding=0, bias=False)
        self.a_proj = Audio_Proj()
        self.gap = nn.AdaptiveAvgPool2d(1)

        ckpt = torch.load("pretrained/PANNs.pth", map_location='cpu')
        self.a_encoder.load_state_dict(ckpt, strict=True)
        for p in self.a_encoder.parameters():
            p.requires_grad = False

        self.tp = tp
        self.tn = tn
        self.ts = 0.03
        self.tc = 0.07

    def forward(self, v_inputs, a_inputs, mode='train'):
        if self.a_encoder.training == True:
            self.a_encoder.eval()

        ea = self.a_encoder(a_inputs).squeeze()
        fa = self.a_proj(ea) # B C
        fa_norm = F.normalize(fa, dim=1, p=2)
        fa_map_norm = fa_norm.unsqueeze(-1).unsqueeze(-1) # B C 1 1
        
        ev = self.v_encoder(v_inputs)
        fv_map = self.v_proj(ev)
        ind_vec = self.gap(fv_map).squeeze()
        fv_map_norm = F.normalize(fv_map, dim=1, p=2) # B C H W
        ind_vec_norm = F.normalize(ind_vec, dim=1, p=2).unsqueeze(-1).unsqueeze(-1)

        if mode == 'train':
            B, C, H, W = fv_map.shape

            tp = int(H * W * self.tp)
            tn = int(H * W * self.tn)
            fv_map_norm_t = rearrange(fv_map_norm, 'b c h w -> h w b c') # H W B C
            ind_vec_norm_t = rearrange(ind_vec_norm, 'b c h w -> h w c b') # 1 1 C B
            Sij = torch.matmul(fv_map_norm_t, ind_vec_norm_t) # H W B B
            Sij = rearrange(Sij, 'h w b c -> b c h w') # B B H W
            Sij = Sij.view(B, B, -1)
            pos_thr = Sij.topk(tp, dim=2)[0][:,:,-1].unsqueeze(-1).repeat(1,1,H*W).view(B,B,H,W)
            neg_thr = Sij.topk(tn, dim=2, largest=False)[0][:,:,-1].unsqueeze(-1).repeat(1,1,H*W).view(B,B,H,W)
            Sij = Sij.view(B, B, H, W)
            m_p = torch.sigmoid((Sij - pos_thr) / self.ts)
            m_n = 1 - torch.sigmoid((Sij - neg_thr) / self.ts)
            SP = (Sij * m_p).sum((2,3)) / m_p.sum((2,3))
            SN = (Sij * m_n).sum((2,3)) / m_n.sum((2,3))
            logits1 = torch.hstack([SP, SN]) / self.tc
            logits2 = torch.hstack([SP.T, SN.T]) / self.tc
            labels = torch.arange(B).cuda()
            loss1 = F.cross_entropy(logits1, labels)
            loss2 = F.cross_entropy(logits2, labels)

            mask = (torch.zeros(B, B) + torch.eye(B) - (1 - torch.eye(B)) / (B - 1)).cuda()
            dist_mat = []
            for i in range(B):
                dist_mat.append(F.pairwise_distance(ind_vec[i].unsqueeze(0).detach(), fa))
            dist_mat = torch.stack(dist_mat) ** 2

            ind_vec_norm = ind_vec_norm.squeeze().detach()
            vv_logits = torch.mm(ind_vec_norm, ind_vec_norm.T)
            w = - vv_logits * (1 - torch.eye(B).cuda()) + torch.eye(B).cuda()
            dist_mat = dist_mat * w

            loss3 = torch.clamp((dist_mat * mask).sum(1) + 0.6, min=0).mean()
            loss4 = torch.clamp((dist_mat * mask).sum(0) + 0.6, min=0).mean()

            return (loss1 + loss2) / 2, (loss3 + loss4) / 2

        else:
            avmap = (fa_map_norm * fv_map_norm).sum(1) # B, H, W
            return avmap
