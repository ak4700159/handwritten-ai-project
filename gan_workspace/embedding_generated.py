import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from your_font_dataset import FontDataset  # 사용자 정의 데이터셋
from your_embedding_model import EmbeddingModel  # 사용자 정의 임베딩 모델

# 데이터 준비
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = FontDataset(root_dir='path/to/font/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 정의 및 훈련
model = EmbeddingModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        images, _ = batch
        
        # Forward pass
        embeddings = model(images)
        loss = criterion(embeddings, images)  # 예: 재구성 손실
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 임베딩 생성
all_embeddings = []
with torch.no_grad():
    for images, _ in dataloader:
        embeddings = model.encoder(images)
        all_embeddings.append(embeddings)

all_embeddings = torch.cat(all_embeddings, dim=0)

# 임베딩 저장
torch.save(all_embeddings, 'EMBEDDINGS.pkl')