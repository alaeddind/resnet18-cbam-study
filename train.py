"""Train one model/dataset/seed; choose best checkpoint by validation loss."""
import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from config import ExperimentConfig
from datasets import create_dataloaders, seed_everything
from models.resnet18 import create_resnet18, VARIANTS
from evaluate import evaluate
from metrics import count_parameters, compute_cost, measure_latency

def train(config, variant="baseline", *, run_test=False):
    config.create_output_dirs()
    seed_everything(config.seed)
    data = create_dataloaders(config)
    # Dataset construction may touch RNG; reset before model initialization.
    seed_everything(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_resnet18(data.num_classes, variant=variant).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    run_name = f"{config.dataset}_{variant}_seed{config.seed}"
    history_path = Path(config.results_dir) / f"{run_name}_history.csv"
    ckpt_path = Path(config.checkpoint_dir) / f"{run_name}_best.pt"
    best_loss = float("inf")
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "lr"])
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss, correct, seen = 0.0, 0, 0
            for x, y in tqdm(data.train_loader, desc=f"{run_name} epoch {epoch}/{config.epochs}", leave=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                seen += y.size(0)
            validation = evaluate(model, data.val_loader, device, criterion)
            row = {"epoch": epoch, "train_loss": total_loss / seen,
                   "train_accuracy": correct / seen, "val_loss": validation["loss"],
                   "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
                   "lr": optimizer.param_groups[0]["lr"]}
            writer.writerow(row)
            file.flush()
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                torch.save({"model_state": model.state_dict(), "config": asdict(config),
                            "variant": variant, "num_classes": data.num_classes,
                            "epoch": epoch, "val_loss": best_loss}, ckpt_path)
            print(f"epoch={epoch} train_acc={row['train_accuracy']:.4f} val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}")
            scheduler.step()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    result = {"dataset": config.dataset, "variant": variant, "seed": config.seed,
              "best_epoch": checkpoint["epoch"], "best_val_loss": best_loss,
              "parameters": count_parameters(model), "device": device,
              "image_size": config.image_size, "checkpoint": str(ckpt_path)}
    result.update(compute_cost(model, config.image_size, device))
    result["latency_ms_batch1"] = measure_latency(model, config.image_size, device)
    if run_test:
        result["test"] = evaluate(model, data.test_loader, device, criterion, return_confusion=True)
    result_path = Path(config.results_dir) / f"{run_name}_summary.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "eurosat"], default="cifar10")
    parser.add_argument("--model", choices=VARIANTS, default="baseline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test", action="store_true", help="Evaluate test set after selecting best validation checkpoint")
    args = parser.parse_args()
    cfg = ExperimentConfig(dataset=args.dataset, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, seed=args.seed, image_size=args.image_size, num_workers=args.num_workers)
    train(cfg, args.model, run_test=args.test)

if __name__ == "__main__":
    main()
