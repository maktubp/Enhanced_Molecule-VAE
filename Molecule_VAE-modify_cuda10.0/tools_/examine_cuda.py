import torch
print(torch.version.cuda)   # 看是不是 8.0
print(torch.cuda.is_available())  # 看是不是 True
print(torch.cuda.get_device_name(0)) # 看你的显卡