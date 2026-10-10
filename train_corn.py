from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("")

    model.train(
        data = "",
        epochs = 100,
        imgsz = 640,
        batch = 8,
        cache = False,
        workers = 0,
        patience = 0,
        device = 0,
        amp = True,

        name = "",

        # 统一26 MuSGD default参数
        optimizer = "MuSGD",
        lr0 = 0.01,
        momentum = 0.9,
        weight_decay = 0.0005,

    )
