import torch
import os
from gan_workspace.common.dataset import TrainDataProvider
from gan_workspace.common.models import Encoder, Decoder, Discriminator
from gan_workspace.common.train import Trainer

# 해당 함수 생성을 위해선 GPU가 있는 환경에서 돌릴 것
# 예상 학습 시간 24시간? -> 어디서 돌릴지 고민 중..
def main():
    # 하이퍼파라미터 및 설정
    GPU = torch.cuda.is_available()
    data_dir = "path/to/your/data"
    fixed_dir = "path/to/fixed/samples"
    fonts_num = 100  # 폰트 수에 맞게 조정
    batch_size = 64
    img_size = 128
    max_epoch = 100
    schedule = 20
    save_path = "path/to/save/results"
    to_model_path = "path/to/save/models"
    lr = 0.001

    # Trainer 인스턴스 생성
    # fixed_dir은 무엇을 의미하는지 모르겠음
    trainer = Trainer(GPU, data_dir, fixed_dir, fonts_num, batch_size, img_size)

    # 학습 실행
    l1_losses, const_losses, category_losses, d_losses, g_losses = trainer.train(
        max_epoch=max_epoch,
        schedule=schedule,
        save_path=save_path,
        to_model_path=to_model_path,
        lr=lr,
        log_step=100,
        sample_step=350,
        fine_tune=False,
        flip_labels=False,
        restore=None,
        from_model_path=None,
        with_charid=False,
        freeze_encoder=False,
        save_nrow=8,
        model_save_step=5,
        resize_fix=90
    )

    # 학습 결과 출력
    print("Training completed.")
    print(f"Final L1 loss: {l1_losses[-1]}")
    print(f"Final Constant loss: {const_losses[-1]}")
    print(f"Final Category loss: {category_losses[-1]}")
    print(f"Final Discriminator loss: {d_losses[-1]}")
    print(f"Final Generator loss: {g_losses[-1]}")

    # 선택적: 학습된 모델을 사용한 추가 작업 (예: 보간)
    # interpolation_results = interpolation(...)

if __name__ == "__main__":
    main()