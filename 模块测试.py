#
# model = YOLO("").model
#
# params = sum(p.numel() for p in model.parameters())
# print(params)

import torch

print("PyTorch:", torch.__version__)
print("CUDA(build):", torch.version.cuda)
print("GPU可用:", torch.cuda.is_available())
