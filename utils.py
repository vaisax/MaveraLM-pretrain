import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def analyze_model_performance(checkpoint_dir="nordic_4b_checkpoints"):
    checkpoint_file = Path(checkpoint_dir) / "checkpoint.pt"
    if not checkpoint_file.exists():
        logger.error("No checkpoint found for analysis")
        return
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    if not train_losses or not val_losses:
        logger.error("No loss data found in checkpoint")
        return
    print(f"\nTraining Analysis:")
    print(f"Training steps completed: {len(train_losses) * 2000}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {min(val_losses):.4f}")
    if len(val_losses) > 10:
        early_loss = np.mean(val_losses[:10])
        recent_loss = np.mean(val_losses[-10:])
        improvement = ((early_loss - recent_loss) / early_loss) * 100
        print(f"Validation loss improvement: {improvement:.2f}%")
    try:
        plt.figure(figsize=(10, 6))
        steps = [i * 2000 for i in range(len(train_losses))]
        plt.plot(steps, train_losses, label='Training Loss', alpha=0.8)
        plt.plot(steps, val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Nordic 4B Model - Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(Path(checkpoint_dir) / 'analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training curve saved to {checkpoint_dir}/analysis.png")
    except Exception as e:
        print(f"Could not create plot: {e}")