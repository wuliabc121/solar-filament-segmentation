"""The original 30 trial settings and 70 extensions used in the M4 E5 experiments."""
from collections import defaultdict
import math
import random

import torch
from torch.nn import functional as F

from .model import TinyUNet, segmentation_loss

SEED = 20261109
SIZES = {384:4,512:2,768:1,1152:1}

DEFAULT = {"width":16,"architecture":"direct","prior":0.,"loss":"base","aug":"none",
           "lr":.001,"schedule":"constant","weight_decay":.0001,"weights":"uniform"}
OPTIONS = (
    ("结构","原 TinyUNet-16",{}),
    ("结构","轻量 TinyUNet-12",{"width":12}),
    ("结构","加宽 TinyUNet-24",{"width":24}),
    ("结构","弱候选残差",{"architecture":"residual","prior":1.}),
    ("结构","强候选残差",{"architecture":"residual","prior":2.}),
    ("损失","双倍 Dice",{"loss":"dice2"}),
    ("损失","Soft-IoU",{"loss":"soft_iou"}),
    ("损失","偏重抑制假阳性的 Tversky",{"loss":"tversky"}),
    ("损失","边缘加权 BCE",{"loss":"boundary"}),
    ("损失","保留候选中已对齐的核心",{"loss":"keep"}),
    ("增强","同步水平或垂直翻转",{"aug":"flip"}),
    ("增强","同步 90 度旋转",{"aug":"rotate"}),
    ("增强","同步 D4",{"aug":"d4"}),
    ("增强","同步 D4 和灰度增强",{"aug":"photo"}),
    ("增强","同步 D4 和语义概率标定扰动",{"aug":"semantic"}),
    ("优化","低学习率 0.0003",{"lr":.0003}),
    ("优化","高学习率 0.002",{"lr":.002}),
    ("优化","余弦衰减",{"schedule":"cosine"}),
    ("优化","预热加余弦衰减",{"schedule":"warm_cosine"}),
    ("优化","较强权重衰减 0.01",{"weight_decay":.01}),
    ("样本权重","粗掩膜低 IoU 加权",{"weights":"hard"}),
    ("样本权重","粗掩膜高 IoU 加权",{"weights":"clean"}),
    ("样本权重","按图像均衡贡献",{"weights":"image"}),
    ("样本权重","小目标加权",{"weights":"small"}),
    ("样本权重","接近 PQ 匹配阈值的候选加权",{"weights":"near"}),
    ("组合","强残差、D4 灰度、余弦",{"architecture":"residual","prior":2.,"aug":"photo","schedule":"cosine"}),
    ("组合","弱残差、边缘损失、余弦",{"architecture":"residual","prior":1.,"loss":"boundary","schedule":"cosine"}),
    ("组合","残差、Tversky、同步翻转",{"architecture":"residual","prior":1.5,"loss":"tversky","aug":"flip"}),
    ("组合","低 IoU 加权、D4、预热余弦",{"weights":"hard","aug":"d4","schedule":"warm_cosine"}),
    ("组合","轻量、按图均衡、D4 灰度",{"width":12,"weights":"image","aug":"photo"}),
)
TRIALS = {f"t{i:02d}":{"direction":direction,"name":name,"config":dict(DEFAULT,**overrides)}
          for i,(direction,name,overrides) in enumerate(OPTIONS)}
RULES = {"direct50":(.5,0.),"direct35":(.35,0.),"direct65":(.65,0.),
         "coarse20":(.5,.2),"coarse40":(.5,.4),"coarse45":(.5,.45)}


class Residual(torch.nn.Module):
    def __init__(self,width,prior):
        super().__init__()
        self.net = TinyUNet(width)
        self.prior = prior

    def forward(self,x):
        return self.net(x)+self.prior*(2*x[:,2:3]-1)


def prior_model(config):
    if config["architecture"]=="residual":
        return Residual(config["width"],config["prior"])
    if config["architecture"] != "direct":
        raise ValueError("Unknown architecture")
    return TinyUNet(config["width"])


def transform(x,y,mode,seed):
    if mode=="none":
        return x,y
    rng = random.Random(seed)
    if mode!="flip":
        k = rng.randrange(4)
        x,y = torch.rot90(x,k,(-2,-1)),torch.rot90(y,k,(-2,-1))
    if mode=="flip":
        axis = -1 if rng.randrange(2) else -2
        x,y = x.flip((axis,)),y.flip((axis,))
    elif mode!="rotate" and rng.randrange(2):
        x,y = x.flip((-1,)),y.flip((-1,))
    if mode=="photo":
        x = torch.cat(((x[:,:1].pow(rng.uniform(.8,1.25))*rng.uniform(.9,1.1)).clamp(0,1),x[:,1:]),dim=1)
    if mode=="semantic":
        x = torch.cat((x[:,:1],x[:,1:2].pow(rng.uniform(.75,1.33)),x[:,2:]),dim=1)
    return x,y


def criterion(logits,target,coarse,mode,weights):
    logits = logits.float()
    dims = (1,2,3)
    if mode=="base" and weights is None:
        return segmentation_loss(logits,target)
    count = target.sum().detach()
    balance = ((target.numel()-count)/count.clamp_min(1)).clamp(1,25)
    bce_map = F.binary_cross_entropy_with_logits(logits,target,pos_weight=balance,reduction="none")
    if mode=="boundary":
        edge = F.max_pool2d(target,5,stride=1,padding=2)+F.max_pool2d(-target,5,stride=1,padding=2)
        emphasized = 1+2*edge
        bce = (bce_map*emphasized).sum(dims)/emphasized.sum(dims)
    else:
        bce = bce_map.mean(dims)
    p = logits.sigmoid()
    inter = (p*target).sum(dims)
    fp = (p*(1-target)).sum(dims)
    fn = ((1-p)*target).sum(dims)
    dice = 1-(2*inter+1)/(2*inter+fp+fn+1)
    if mode=="soft_iou":
        overlap = 1-(inter+1)/(inter+fp+fn+1)
    elif mode=="tversky":
        overlap = 1-(inter+1)/(inter+.65*fp+.35*fn+1)
    else:
        overlap = (2 if mode=="dice2" else 1)*dice
    loss = bce+overlap
    if mode=="keep":
        core = coarse*target
        loss += .15*(F.softplus(-logits)*core).sum(dims)/core.sum(dims).clamp_min(1)
    return (loss*weights).sum()/weights.sum() if weights is not None else loss.mean()


def step_weights(mode,rows,y,images):
    if mode=="uniform":
        return None
    if mode=="small":
        return (48/(y.sum((1,2,3)).clamp_min(1).sqrt())).clamp(.5,2.)
    if mode=="image":
        values = [images[r["observation_id"]]**-.5 for r in rows]
    else:
        ious = [r["candidate_iou"] for r in rows]
        if mode=="hard":
            values = [1.5-v for v in ious]
        elif mode=="clean":
            values = [.5+v for v in ious]
        else:
            values = [.5+math.exp(-abs(v-.5)/.12) for v in ious]
    return torch.tensor(values,dtype=torch.float32,device=y.device)


def learning_rate(config,step,total):
    base = config["lr"]
    policy = config["schedule"]
    if policy=="constant":
        return base
    warm = max(1,int(.05*total)) if policy=="warm_cosine" else 0
    if step<warm:
        return base*(.1+.9*(step+1)/warm)
    phase = (step-warm)/max(1,total-warm-1)
    return base*(.1+.45*(1+math.cos(math.pi*phase)))


def shuffled_batches(records,seed):
    buckets = defaultdict(list)
    for i,(_,_,row) in enumerate(records):
        buckets[row["transform"]["crop_hw"][0]].append(i)
    for epoch in range(4):
        rng = random.Random(seed+epoch)
        batches = []
        for side,indices in sorted(buckets.items()):
            order = rng.sample(indices,len(indices))
            size = SIZES[side]
            batches.extend((side,order[i:i+size]) for i in range(0,len(order),size))
        rng.shuffle(batches)
        yield from batches


class SeparateEditHeads(TinyUNet):
    def __init__(self,width):
        super().__init__(width)
        original = self.head
        self.head = torch.nn.Conv2d(width,2,1)
        with torch.no_grad():
            self.head.weight.copy_(original.weight.expand_as(self.head.weight))
            self.head.bias.copy_(original.bias.expand_as(self.head.bias))

    def forward(self,x):
        logits = super().forward(x)
        return torch.where(x[:,2:3]>=.5,logits[:,:1],logits[:,1:])


EXTRA = (
    *[("结构",f"TinyUNet-{w}",{"width":w}) for w in (8,20,28,32)],
    *[("结构",f"候选残差 {p}",{"architecture":"residual","prior":p}) for p in (.5,1.5,2.5,3.)],
    *[("结构",f"保留和新增双头-{w}",{"architecture":"separate_heads","width":w}) for w in (12,16)],
    *[("损失",f"Dice 系数 {w}",{"loss":"dice_tuned","loss_strength":w}) for w in (.5,1.5,2.5)],
    *[("损失",f"Tversky FP 权重 {w}",{"loss":"tversky_tuned","loss_strength":w}) for w in (.55,.75,.85)],
    *[("损失",f"边缘额外权重 {w}",{"loss":"boundary_tuned","loss_strength":w}) for w in (.5,1.)],
    *[("损失",f"IoU 下限 0.55，惩罚 {w}",{"loss":"iou_tuned","loss_strength":w,"iou_target":.55}) for w in (.25,.5)],
    *[("增强",f"D4 灰度扰动 {w}",{"aug":"photo_tuned","aug_strength":w}) for w in (.06,.15,.35)],
    *[("增强",f"D4 语义概率扰动 {w}",{"aug":"semantic_tuned","aug_strength":w}) for w in (.08,.18,.4)],
    *[("增强",f"成对翻转概率 {w}",{"aug":"flip_tuned","aug_strength":w}) for w in (.25,.75)],
    *[("增强",f"同步 180 度旋转，灰度扰动 {w}",{"aug":"rotate180","aug_strength":w}) for w in (0.,.18)],
    *[("优化",f"恒定学习率 {w}",{"lr":w}) for w in (.0005,.0007,.0013,.0016)],
    *[("优化",f"权重衰减 {w}",{"weight_decay":w}) for w in (.001,.003,.03)],
    *[("优化",f"余弦调度起点 {w}",{"schedule":"cosine","lr":w}) for w in (.0007,.0015)],
    ("优化","预热余弦起点 0.0007",{"schedule":"warm_cosine","lr":.0007}),
    *[("样本权重",f"低 IoU 加权强度 {w}",{"weights":"hard_tuned","weight_strength":w}) for w in (.25,.5,1.5)],
    *[("样本权重",f"高 IoU 加权强度 {w}",{"weights":"clean_tuned","weight_strength":w}) for w in (.5,1.5)],
    *[("样本权重",f"图像均衡指数 {w}",{"weights":"image_tuned","weight_strength":w}) for w in (.25,.75)],
    *[("样本权重",f"小目标权重尺度 {w}",{"weights":"small_tuned","weight_strength":w}) for w in (24,36,64)],
    ("组合","宽度 20 与双倍 Dice",{"width":20,"loss":"dice2"}),
    ("组合","宽度 24 与余弦衰减",{"width":24,"schedule":"cosine"}),
    ("组合","残差 0.5 与 Dice 系数 1.5",{"architecture":"residual","prior":.5,"loss":"dice_tuned","loss_strength":1.5}),
    ("组合","残差 1.5 与 Tversky",{"architecture":"residual","prior":1.5,"loss":"tversky_tuned","loss_strength":.55}),
    ("组合","双头与小目标加权",{"architecture":"separate_heads","weights":"small"}),
    ("组合","低学习率与 D4",{"lr":.0005,"aug":"d4"}),
    ("组合","D4 灰度增强与权重衰减",{"aug":"photo","weight_decay":.003}),
    ("组合","按图均衡与余弦",{"weights":"image","schedule":"cosine"}),
    ("组合","小目标加权与双倍 Dice",{"weights":"small","loss":"dice2"}),
    ("组合","边缘损失与小目标加权",{"weights":"small","loss":"boundary"}),
    *[("纠错监督",f"IoU 下限 {w}",{"loss":"iou_tuned","loss_strength":.5,"iou_target":w}) for w in (.45,.50,.60,.65)],
    *[("纠错监督",f"差异像素额外权重 {w}",{"loss":"edit_tuned","loss_strength":w}) for w in (.1,.25,.5)],
    *[("纠错监督",f"共同前景核心惩罚 {w}",{"loss":"keep_tuned","loss_strength":w}) for w in (.05,.3,.5)],
)
NEW = {f"t{i:02d}":{"direction":direction,"name":name,"config":dict(DEFAULT,**config)}
       for i,(direction,name,config) in enumerate(EXTRA,30)}
TRIALS = {**TRIALS,**NEW}


def make_model(config):
    if config["architecture"]=="separate_heads":
        return SeparateEditHeads(config["width"])
    return prior_model(config)


def augment(x,y,config,seed):
    mode = config["aug"]
    if "aug_strength" not in config:
        return transform(x,y,mode,seed)
    strength = config["aug_strength"]
    rng = random.Random(seed+913)
    if mode in ("photo_tuned","semantic_tuned"):
        x,y = transform(x,y,"d4",seed)
        channel = 0 if mode=="photo_tuned" else 1
        value = x[:,channel:channel+1].pow(math.exp(rng.uniform(-strength,strength)))
        return torch.cat((x[:,:channel],value,x[:,channel+1:]),dim=1),y
    if mode=="flip_tuned":
        if rng.random()<strength:
            return transform(x,y,"flip",seed)
        return x,y
    if mode != "rotate180":
        raise ValueError("Unknown augmentation")
    if rng.randrange(2):
        x,y = torch.rot90(x,2,(-2,-1)),torch.rot90(y,2,(-2,-1))
    if strength:
        gray = x[:,:1].pow(math.exp(rng.uniform(-strength,strength)))
        x = torch.cat((gray,x[:,1:]),dim=1)
    return x,y


def sample_weight(cfg,rows,y,images):
    mode = cfg["weights"]
    if "weight_strength" not in cfg:
        return step_weights(mode,rows,y,images)
    value = cfg["weight_strength"]
    if mode=="small_tuned":
        return (value/y.sum((1,2,3)).clamp_min(1).sqrt()).clamp(.5,2.)
    if mode=="image_tuned":
        weights = [images[r["observation_id"]]**-value for r in rows]
    elif mode=="hard_tuned":
        weights = [1+value*(1-r["candidate_iou"]) for r in rows]
    else:
        if mode != "clean_tuned":
            raise ValueError("Unknown weighting rule")
        weights = [1+value*r["candidate_iou"] for r in rows]
    return torch.tensor(weights,dtype=torch.float32,device=y.device)


def objective(logits,target,coarse,cfg,weights):
    mode = cfg["loss"]
    if "loss_strength" not in cfg:
        return criterion(logits,target,coarse,mode,weights)
    z = logits.float()
    p = z.sigmoid()
    dims = (1,2,3)
    strength = cfg["loss_strength"]
    count = target.sum().detach()
    balance = ((target.numel()-count)/count.clamp_min(1)).clamp(1,25)
    loss_map = F.binary_cross_entropy_with_logits(z,target,pos_weight=balance,reduction="none")
    if mode=="boundary_tuned":
        edge = F.max_pool2d(target,5,stride=1,padding=2)+F.max_pool2d(-target,5,stride=1,padding=2)
        emphasis = 1+strength*edge
        bce = (loss_map*emphasis).sum(dims)/emphasis.sum(dims)
    elif mode=="edit_tuned":
        emphasis = 1+strength*(coarse>=.5).ne(target>=.5).float()
        bce = (loss_map*emphasis).sum(dims)/emphasis.sum(dims)
    else:
        bce = loss_map.mean(dims)
    intersection = (p*target).sum(dims)
    p_sum,y_sum = p.sum(dims),target.sum(dims)
    dice = 1-(2*intersection+1)/(p_sum+y_sum+1)
    if mode=="tversky_tuned":
        overlap = 1-(intersection+1)/(intersection+strength*(p_sum-intersection)+(1-strength)*(y_sum-intersection)+1)
    elif mode=="dice_tuned":
        overlap = strength*dice
    else:
        overlap = dice
    if mode=="iou_tuned":
        iou = (intersection+1)/(p_sum+y_sum-intersection+1)
        overlap = overlap+strength*(cfg["iou_target"]-iou).clamp_min(0)
    if mode=="keep_tuned":
        core = coarse*target
        overlap = overlap+strength*(F.softplus(-z)*core).sum(dims)/core.sum(dims).clamp_min(1)
    loss = bce+overlap
    return loss.mean() if weights is None else (loss*weights).sum()/weights.sum()
