import torch
import os
from common.dataset import TrainDataProvider
from common.models import Encoder, Decoder, Discriminator
from common.train import Trainer
from embedding_generated import generate_font_embeddings

# 해당 함수 생성을 위해선 GPU가 있는 환경에서 돌릴 것
# 예상 학습 시간 24시간? -> 어디서 돌릴지 고민 중..
def main():
    embeddings = generate_font_embeddings(
    fonts_num=25,
    embedding_dim=128,
    save_dir="./fixed_dir",
    stddev=0.01
)

if __name__ == "__main__":
    main()