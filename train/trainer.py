import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from .metrics import compute_iou


def train(
    model: nn.Module, train_loader, val_loader, criterion,
    optimizer, num_epochs=100, patience=10,  # patience=t
    save_path="model_weights/best_model.pth",
    deep_supervision=True,
    device='cpu',
    pre_trained_weights=None
):
    if pre_trained_weights is not None:
        model.load_state_dict(pre_trained_weights)
    best_iou = 0.0
    epochs_no_improve = 0
    history_weights = []  # list buffer to store weights for rollback

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if deep_supervision:
                loss = criterion((outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4, masks)
            else:
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ====== 验证阶段，计算IoU ======
        model.eval()
        total_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                preds = model(images)
                if deep_supervision:
                    batch_iou = compute_iou((preds[0] + preds[1] + preds[2] + preds[3]) / 4, masks)
                else:
                    batch_iou = compute_iou(preds, masks)
                total_iou += batch_iou

        avg_val_iou = total_iou / len(val_loader)
        print(f"[Epoch {epoch}] Loss={avg_train_loss:.4f}  IoU={avg_val_iou:.4f}")

        # ====== 保存当前模型权重以备回滚 ======
        history_weights.append(model.state_dict())
        history_weights = history_weights[-patience:]

        # ====== 监测性能提升 ======
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            epochs_no_improve = 0
            print("  -> IoU improved.")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

        # ====== 检测是否触发回滚 ======
        if epochs_no_improve >= patience:
            rollback_epoch = epoch - patience
            target_state_dict = history_weights[-patience]
            torch.save(target_state_dict, save_path)
            print(f"*** Performance dropped for {patience} epochs → "
                  f"rollback and save weights of epoch {rollback_epoch} to {save_path} ***")
            return  # 结束训练（可改为继续）

    # 如果训练期间没有回滚也会保存最佳模型
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Best IoU model saved to {save_path}.")
