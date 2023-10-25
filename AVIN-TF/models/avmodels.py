import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vision_transformer import vit_small
from einops import rearrange
from scipy.linalg import eigh
from scipy import ndimage
import numpy as np
from models.audnet import ResNet22

class Visual_Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

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
        self.v_encoder = vit_small(pretrained="pretrained/dino_deitsmall16_pretrain.pth")
        self.a_encoder = ResNet22()
        self.v_proj = Visual_Proj()
        self.a_proj = Audio_Proj()

        for p in self.v_encoder.parameters():
            p.requires_grad = False
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
        if self.v_encoder.training == True:
            self.v_encoder.eval()

        ea = self.a_encoder(a_inputs).squeeze()
        fa = self.a_proj(ea) # B C
        fa_norm = F.normalize(fa, dim=1, p=2)
        fa_map_norm = fa_norm.unsqueeze(-1).unsqueeze(-1) # B C 1 1

        ev = self.v_encoder.get_last_key(v_inputs) # B H 1+H*W C
        ev = rearrange(ev[:,:,1:], 'B h (H W) c -> B (h c) H W', H=14) # B C H W
        fv_map = self.v_proj(ev)
        fv_map_norm = F.normalize(fv_map, dim=1, p=2) # B C H W

        if mode == 'train':
            B, C, H, W = fv_map.shape

            masks = []
            ev_Ncut = F.normalize(ev.flatten(2), p=2, dim=1).detach()
            for feat in ev_Ncut:
                masks.append(detect_mask(feat, (14,14), 0.2))
            masks = torch.from_numpy(np.array(masks)).cuda().float()
            ind_vec = (fv_map * masks.unsqueeze(1)).sum((2,3)) / (masks.sum((1,2)).unsqueeze(-1)) # B C
            ind_vec_norm = F.normalize(ind_vec, dim=1, p=2).unsqueeze(-1).unsqueeze(-1)

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

def detect_mask(feats, dims, tau, eps=1e-5):
    A = (feats.transpose(0,1) @ feats)
    A = A.cpu().numpy()
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    seed = np.argmax(np.abs(second_smallest_vec))
    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]
    cc = np.where(objects == cc)
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1
    return mask