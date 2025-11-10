import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from reflecto.dataset import DatasetH5, ParamQuantizer
from reflecto.model import XRRClassifier, train_epoch, validate_epoch


def main():
    # ---------------------------------------------------
    # 기본 설정
    # ---------------------------------------------------
    h5_path = r"D:\03_Resources\Data\XRR_AI\data\p300o6_raw.h5"
    model_path = r"xrr_model2.pt"
    batch_size = 32
    epochs = 20
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------
    # 1. Quantizer 생성
    # ---------------------------------------------------
    quantizer = ParamQuantizer(
        thickness_bins=None,
        roughness_bins=None,
        sld_bins=None
    )

    n_th = len(quantizer.thickness_bins) - 1
    n_rg = len(quantizer.roughness_bins) - 1
    n_sld = len(quantizer.sld_bins) - 1

    # ---------------------------------------------------
    # 2. Dataset / DataLoader 구성
    # ---------------------------------------------------
    train_dataset = DatasetH5(h5_path, quantizer)
    val_dataset   = DatasetH5(h5_path, quantizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ---------------------------------------------------
    # 3. reflectivity 길이 자동 감지
    # ---------------------------------------------------
    sample_refl, sample_label = train_dataset[0]
    q_len = sample_refl.shape[0]
    n_layers = sample_label.shape[0]
    print(f"[INFO] q_len={q_len}, layers={n_layers}")

    # ---------------------------------------------------
    # 4. 모델 생성 + compile
    # ---------------------------------------------------
    model = XRRClassifier(
        q_len=q_len,
        n_layers=n_layers,
        n_th_bins=n_th,
        n_rg_bins=n_rg,
        n_sld_bins=n_sld
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda")  # AMP 안정성

    # ---------------------------------------------------
    # 5. 학습 루프
    # ---------------------------------------------------
    for epoch in range(1, epochs + 1):

        train_loss, train_parts = train_epoch(
            model, train_loader, optimizer, device, scaler
        )

        val_loss, val_accs = validate_epoch(
            model, val_loader, device
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc(th/rg/sld): "
            f"{val_accs['th']:.3f} / {val_accs['rg']:.3f} / {val_accs['sld']:.3f}"
        )

    # ---------------------------------------------------
    # 6. 모델 + quantizer 저장
    # ---------------------------------------------------
    ckpt = {
        "model": model.state_dict(),
        "quantizer": quantizer.state_dict()
    }
    torch.save(ckpt, model_path)
    print(f"[INFO] Saved: {model_path}")


if __name__ == "__main__":
    main()
