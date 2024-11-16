import os
import torch
from common.train import Trainer
from common.function import init_embedding, generate_font_embeddings

def main():
    # GPU 사용 가능 여부 확인
    GPU = torch.cuda.is_available()
    
    # 기본 설정값
    data_dir = "./data"  # 학습 데이터가 있는 디렉토리
    fixed_dir = "./fixed_dir"  # 고정 데이터(embeddings 등)가 저장될 디렉토리
    save_path = "./samples"  # 생성된 이미지 샘플이 저장될 경로
    model_path = "./models"  # 학습된 모델이 저장될 경로
    
    # 하이퍼파라미터 설정
    fonts_num = 10  # 학습할 폰트 개수
    batch_size = 16
    img_size = 128
    max_epoch = 100
    schedule = 20  # learning rate 조정 주기
    embedding_dim = 128
    
    # 필요한 디렉토리 생성
    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # EMBEDDINGS.pkl 파일이 없을 경우 생성
    embedding_path = os.path.join(fixed_dir, 'EMBEDDINGS.pkl')
    if not os.path.exists(embedding_path):
        print("Generating new embeddings...")
        generate_font_embeddings(
            fonts_num=fonts_num,
            embedding_dim=embedding_dim,
            save_dir=fixed_dir,
            stddev=0.01
        )
    
    # Trainer 인스턴스 생성
    trainer = Trainer(
        GPU=GPU,
        data_dir=data_dir,
        fixed_dir=fixed_dir,
        fonts_num=fonts_num,
        batch_size=batch_size,
        img_size=img_size
    )
    
    # 학습 설정
    train_params = {
        'max_epoch': max_epoch,
        'schedule': schedule,
        'save_path': save_path,
        'to_model_path': model_path,
        'lr': 0.001,
        'log_step': 100,  # 로그 출력 주기
        'sample_step': 350,  # 샘플 이미지 생성 주기
        'model_save_step': 5,  # 모델 저장 주기(epoch 단위)
        'fine_tune': False,  # 미세 조정 여부
        'with_charid': True,  # 문자 ID 사용 여부
        'flip_labels': False,  # 레이블 플립 사용 여부
        'save_nrow': 8,  # 저장될 이미지 그리드의 행 개수
        'resize_fix': 90  # 이미지 크기 조정 기준값
    }
    
    # 이전 학습된 모델이 있다면 복원하여 이어서 학습
    restore = None
    if os.path.exists(model_path):
        model_files = sorted([f for f in os.listdir(model_path) if f.endswith('.pkl')])
        if len(model_files) >= 3:  # Encoder, Decoder, Discriminator 모델 파일이 모두 있는 경우
            # 가장 최근 모델 파일들을 찾아서 복원
            latest_epoch = max([int(f.split('-')[0]) for f in model_files])
            restore = [
                f'{latest_epoch}-*-Encoder.pkl',
                f'{latest_epoch}-*-Decoder.pkl',
                f'{latest_epoch}-*-Discriminator.pkl'
            ]
            train_params['from_model_path'] = model_path
            train_params['restore'] = restore
            print(f"Restoring from epoch {latest_epoch}")
    
    # 학습 시작
    print("Starting training...")
    losses = trainer.train(**train_params)
    
    # 손실값 저장
    l1_losses, const_losses, category_losses, d_losses, g_losses = losses
    print("Training completed!")
    
    return losses

if __name__ == "__main__":
    main()