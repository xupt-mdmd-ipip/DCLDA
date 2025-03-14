import numpy as np
import torch

SMALL_NUM = np.log(1e-45)

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def debiased_loss(out_1,out_2,batch_size, temperature, debiased, tau_plus):
    # pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    # feature_1, out_1 = net(pos_1)
    # feature_2, out_2 = net(pos_2)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    if debiased:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e**(-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng))).mean()
    return loss
