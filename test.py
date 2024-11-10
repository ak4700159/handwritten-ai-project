import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# # 공개 데이터셋에서 학습 데이터를 내려받습니다.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# # 공개 데이터셋에서 테스트 데이터를 내려받습니다.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# batch_size = 64

# # 데이터로더를 생성합니다.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)


# for X, y in test_dataloader:
#         # GPU가 존재하면 텐서를 이동합니다
#     if torch.cuda.is_available():
#         tensor = X.to("cuda")
#         tensor = y.to("cuda")

#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(X.device)
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     print(y)
#     break



test = torch.ones([10, 3])
print(test)