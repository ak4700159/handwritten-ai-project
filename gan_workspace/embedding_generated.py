import os
import torch
import torch.nn.functional as F  # F로 자주 임포트되는 PyTorch의 함수형 인터페이스
from common.function import init_embedding

def generate_font_embeddings(fonts_num, embedding_dim=128, save_dir="./", stddev=0.01):
    """
    fonts_num: 학습할 폰트 스타일의 수
    embedding_dim: 임베딩 벡터의 차원
    save_dir: 임베딩 파일을 저장할 경로
    stddev: 정규분포의 표준편차
    """
    # 초기 임베딩 생성 (fonts_num x 1 x 1 x embedding_dim)
    embeddings = init_embedding(fonts_num, embedding_dim, stddev=stddev)
    
    # L2 정규화 적용 - 각 폰트 스타일의 임베딩 벡터를 단위 벡터로 정규화
    embeddings_flat = embeddings.view(fonts_num, -1)
    embeddings_normalized = F.normalize(embeddings_flat, p=2, dim=1)
    embeddings = embeddings_normalized.view(fonts_num, 1, 1, embedding_dim)
    
    # 저장
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'EMBEDDINGS.pkl')
    torch.save(embeddings, save_path)
    
    print(f"Font style embeddings generated and saved to {save_path}")
    print(f"Shape: {embeddings.shape}")
    
    return embeddings