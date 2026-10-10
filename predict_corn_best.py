from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"")

    results = model.val(
        data=r"",  # 配置文件路径
        workers=0,
        split="test",  # 指定使用测试集。
        verbose=True,  # 结果
        name="val",
        conf=0.001,
    )
