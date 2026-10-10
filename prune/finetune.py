from prune import DetectionFinetune

config = {
    # ---------------- model ----------------
    "model": r"D:\yolo26\runs\detect\yolo26n_ghost_width15_prune13\weights\pruned_model_lamp.pt",
    "data": r"D:\yolo26\ultralytics\cfg\datasets\cornleafdiseases.yaml",
    # ---------------- Train ----------------
    "epochs": 1,
    "imgsz": 640,
    "batch": 8,
    "device": 0,
    "workers": 0,
    "cache": False,
    "amp": True,
    # ---------------- save ----------------
    "project": r"D:\yolo26\runs\detect",
    "name": "yolo26n_ghost_width15_prune13_nf",
    # ---------------- 与正式训练保持一致 ----------------
    "optimizer": "MuSGD",
    "lr0": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    # ---------------- 其它 ----------------
    "patience": 0,
}

trainer = DetectionFinetune(overrides=config)
trainer.train()
