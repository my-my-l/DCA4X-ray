from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Optional
import torch
import torch.nn as nn



def auc(output, target):
    with torch.no_grad():
        output_array = output.cpu().numpy()
        target_array = target.cpu().numpy()
        try:
            res  = [roc_auc_score(target_array,output_array,average=None)*100.0]
        except ValueError:
            res = []
    return res

def mix(x_s, x_t, y_s, y_t, device, a=1.0, b=1.0, lam_t=None, get_x_t=False):
    if lam_t is None:
        if a > 0 and b > 0:
            lam_t = np.random.beta(a, b)
        else:
            lam_t = 1
    
    batch_size = x_s.size(0)
    index_t = torch.randperm(batch_size).to(device)
    mixed_x = lam_t * x_t[index_t, :] + (1-lam_t) * x_s 
    y_a, y_b = y_t[index_t], y_s
  
    return (mixed_x, y_a, y_b, lam_t)

class DataGenerator(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = True, sup: Optional[float]=10.):
        super(DataGenerator, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step
        self.coeff = 0
        self.sup = sup

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor, y_s: torch.Tensor, y_t: torch.tensor, device, get_x_t=False) -> torch.Tensor:
        """"""
        coeff = self.sup * np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        self.coeff = coeff
        if self.auto_step:
            self.step()
        return mix(x_s,x_t,y_s,y_t,device,a=coeff, b=self.sup-coeff, get_x_t=get_x_t)

    def step(self):
        self.iter_num += 1

def mixing_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_target_samples(hi_threshold: float, lo_threshold: float, x, y, device, sample_num=None):
    
    y = y.detach()
    score = torch.sigmoid(y)
    indices = torch.nonzero(((score > hi_threshold) | (score < lo_threshold)), as_tuple=False).view(-1)
    indices = indices.repeat(sample_num // indices.size(0) + 1)
    indices = indices[torch.randperm(indices.size()[0]).to(device)]
    indices = indices[:sample_num]
    return x[indices], score[indices]

def get_mask(hi_threshold: float, lo_threshold: float, y):
    y = y.detach()
    y = torch.sigmoid(y)
    mask_p = (y > hi_threshold).float()
    mask_n = (y < lo_threshold).float()
    return mask_p, mask_n
   

class SDLoss(nn.Module):
    def __init__(self):
        super(SDLoss, self).__init__()

    def forward(self, f_1, f_2, f_3=None):
        _, s_1, _ = torch.svd(f_1)
        _, s_2, _ = torch.svd(f_2)
        _, s_3, _ = torch.svd(f_3)
        loss = torch.pow(s_1[0], 2) + torch.pow(s_2[0], 2) + torch.pow(s_3[0], 2)
        return loss
   