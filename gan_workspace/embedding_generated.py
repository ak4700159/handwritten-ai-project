import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# 임베딩 모델 정의
class FontEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim):
        super(FontEmbeddingModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 데이터 로딩 및 전처리
def load_font_data(data_dir):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    dataset = []
    for font_name in os.listdir(data_dir):
        font_path = os.path.join(data_dir, font_name)
        img = Image.open(font_path)
        img_tensor = transform(img)
        dataset.append((img_tensor, font_name))
    
    return DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 학습
def train_model(model, dataloader, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch, _ in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 임베딩 추출 및 저장
def extract_embeddings(model, dataloader):
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch, font_names in dataloader:
            output = model(batch)
            for embedding, font_name in zip(output, font_names):
                embeddings[font_name] = embedding.unsqueeze(0).unsqueeze(0)
    
    embeddings_tensor = torch.cat(list(embeddings.values()), dim=0)
    torch.save(embeddings_tensor, 'EMBEDDINGS.pkl')

# 메인 실행
if __name__ == "__main__":
    data_dir = "path/to/font/images"
    embedding_dim = 128

    dataloader = load_font_data(data_dir)
    model = FontEmbeddingModel(embedding_dim)
    
    train_model(model, dataloader)
    extract_embeddings(model, dataloader)