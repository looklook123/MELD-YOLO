from ultralytics import YOLO

model = YOLO(r"")

model.export(
    format="ncnn",
    imgsz=640,
    # onnx
    # opset=12,
    # simplify=True,
)
