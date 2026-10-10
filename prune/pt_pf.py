import torch

ckpt=torch.load(
    r"D:\yolo26\runs\detect\yolo26n_ghost_width15_prune15\weights\pruned_model_lamp.pt",
    map_location="cpu",
    weights_only=False
)

print(type(ckpt["model"]))

print(sum(p.numel() for p in ckpt["model"].parameters()))