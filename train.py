import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from datasets import make_dataloaders
from models import make_model
from utils.seed import set_seed
from eval import evaluate
from config_loader import load_configs, print_config
import os
from pathlib import Path
import pandas as pd

def train(config):
    set_seed(config.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create results directory
    results_dir = config.get("results_dir", "results")
    Path(results_dir).mkdir(exist_ok=True)

    train_loader, val_loader, num_classes = make_dataloaders(
        name=config["dataset"],
        batch_size=config.get("batch_size", 64)
    )

    model = make_model(
        arch=config["arch"],
        num_classes=num_classes,
        pretrained=config.get("pretrained", False),
        freeze_backbone=config.get("freeze_backbone", False)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 0))
    
    scheduler = None
    if config.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("epochs", 10))

    scaler = GradScaler() if config.get("use_amp", False) else None

    best_acc = 0
    best_metrics = None
    epoch_history = []
    checkpoint_path = os.path.join(results_dir, "best_model.pt")
    
    for epoch in range(config.get("epochs", 10)):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if config.get("use_grad_clip", False):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if config.get("use_grad_clip", False):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()

        if scheduler:
            scheduler.step()

        # Evaluate on validation set
        confusion_matrix_path = os.path.join(results_dir, f"confusion_matrix_epoch_{epoch+1}.png")
        val_metrics = evaluate(model, val_loader, device=device, save_path=confusion_matrix_path)
        val_acc = val_metrics.get("accuracy", 0)
        
        avg_train_loss = train_loss / len(train_loader)
        
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc,
            "metrics": val_metrics
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = val_metrics
            # Save best model checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → Best model saved to {checkpoint_path}")
        
        print(f"Epoch {epoch + 1}/{config.get('epochs', 10)} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
    # Save summary to text file
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_acc:.4f}\n\n")
        
        f.write("Per-Epoch Results:\n")
        f.write("-" * 50 + "\n")
        for record in epoch_history:
            f.write(f"\nEpoch {record['epoch']}:\n")
            f.write(f"  Train Loss: {record['train_loss']:.4f}\n")
            f.write(f"  Val Accuracy: {record['val_accuracy']:.4f}\n")
        
        if best_metrics:
            f.write("\n" + "=" * 50 + "\n")
            f.write("Best Model Metrics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
            if "macro" in best_metrics:
                macro = best_metrics["macro"]
                f.write(f"Macro Precision: {macro[0]:.4f}\n")
                f.write(f"Macro Recall: {macro[1]:.4f}\n")
                f.write(f"Macro F1-Score: {macro[2]:.4f}\n")

    # -----------------------------
    # Part 1: write experiment-level summary to part_1_results
    # -----------------------------
    part1_dir = Path("part_1_results")
    part1_dir.mkdir(exist_ok=True)

    # Load best model for final evaluation (if exists)
    best_ckpt = os.path.join(results_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        eval_model = make_model(
            arch=config["arch"],
            num_classes=num_classes,
            pretrained=config.get("pretrained", False),
            freeze_backbone=config.get("freeze_backbone", False)
        )
        eval_model.load_state_dict(torch.load(best_ckpt, map_location=device))
        eval_model.to(device)

        cm_path = part1_dir / f"cn_{config['arch']}_{config['dataset']}.png"
        final_metrics = evaluate(eval_model, val_loader, device=device, save_path=str(cm_path))

        # Build summary text
        summary_file = part1_dir / "summary.txt"
        with open(summary_file, "a") as sf:
            sf.write(f"Experiment: {config['arch']} | Dataset: {config['dataset']}\n")

            # per-class table
            per_class_df = final_metrics["per_class"]
            per_class_df.index.name = "class"
            sf.write(per_class_df.to_string())
            sf.write("\n\n")

            # global aggregates
            global_df = final_metrics["global"]
            sf.write(global_df.to_string())
            sf.write("\n\n")

        print(f"Part 1: summary appended to {summary_file} and confusion matrix saved to {cm_path}")
    else:
        print(f"No best checkpoint found at {best_ckpt}; skipping part_1 summary append.")

if __name__ == "__main__":
    # Load configuration from YAML files
    config = load_configs(
        model_config_path="configs/model.yaml",
        train_config_path="configs/train.yaml"
    )
    
    # Print loaded configuration
    print_config(config)
    
    # Start training
    train(config)
