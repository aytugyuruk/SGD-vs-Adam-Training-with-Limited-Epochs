# Libraries
from utils.seed import seed_everything
import torch
import os
from torch.utils.data import DataLoader, random_split
from itertools import product
from initialization.build_initialization import initialize_weights
from datasets.mnist import get_mnist_datasets
from models.mlp import MLP
from optimizers.build_optimizer import build_optimizer
from schedulers.build_scheduler import build_scheduler
from training.loops import train_one_epoch, validate_one_epoch

# Device selection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Main code
def run_experiments():
    device = get_device()
    print(f"Active Device: {device}")
    seed_everything(3)

    full_train_dataset = get_mnist_datasets(train=True)
    test_dataset = get_mnist_datasets(train=False)

    train_size = int(0.9 * len(full_train_dataset))
    val_size   = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset,[train_size, val_size],generator=torch.Generator().manual_seed(3))

    train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset,batch_size=128,shuffle=False,num_workers=2)
    test_loader = DataLoader(test_dataset,batch_size=128,shuffle=False,num_workers=2)

    # Experiment space
    initializations = ["xavier", "he"]
    optimizers = ["sgd", "adam"]
    schedulers = ["step", "cosine", "exponential"]
    # Loss function is constant across experiments
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Budget configurations (to compare performance under different training durations)
    budgets = [5, 30]

    os.makedirs("checkpoints", exist_ok=True)

    for budget in budgets:
        print("\n" + "=" * 80)
        print(f"BUDGET: {budget} EPOCHS")
        print("=" * 80 + "\n")
        
        for exp_id, (init, opt, sched) in enumerate(product(initializations, optimizers, schedulers), start=1):
            print(f"Starting training...\n"
                  f"{'= ' * 40}\n"
                  f"Experiment {exp_id} (Budget: {budget})\n"
                  f"Initialization : {init}\n"
                  f"Optimizer      : {opt}\n"
                  f"LR Scheduler   : {sched}\n"
                  f"{'= ' * 40}\n")

            # Model & optimizer
            model = MLP(784, 128, 10).to(device)

            #Assign the parameters
            initialize_weights(model, init)
            optimizer = build_optimizer(model,lr=1e-3,optimizer_type=opt)
            scheduler = build_scheduler(optimizer,lr_type=sched)

            # Training loop
            epoch_losses = []
            val_losses = []
            gap_losses = []

            best_f1 = 0.0
            best_epoch = -1
            best_state = None

            for epoch in range(budget):
                train_loss = train_one_epoch(model,train_loader,loss_fn,optimizer,device)
                scheduler.step()
                epoch_losses.append(train_loss)

                val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch(model,val_loader,loss_fn,device)
                val_losses.append(val_loss)
                
                gap = val_loss - train_loss
                gap_losses.append(gap)

                print(
                    f"Epoch [{epoch+1:02d}/{budget}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Gap: {gap:.4f}"
                )

                # Best model selection (by validation F1)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_epoch = epoch + 1
                    best_state = model.state_dict()

            # Final evaluation on Test set
            model.load_state_dict(best_state)

            test_loss, test_acc, test_prec, test_rec, test_f1 = validate_one_epoch(model,test_loader,loss_fn,device)

            print("-" * 80)
            print(f"Best Epoch (Val) : {best_epoch}")
            print(f"Test Accuracy   : {test_acc:.4f}")
            print(f"Test Precision  : {test_prec:.4f}")
            print(f"Test Recall     : {test_rec:.4f}")
            print(f"Test F1 Score   : {test_f1:.4f}")
            print("-" * 80)

            # Checkpoint
            ckpt_path = (
                f"checkpoints/"
                f"mlp_budget-{budget}_init-{init}_opt-{opt}_sched-{sched}.pth"
            )

            torch.save(
                {
                    "model_state": best_state,
                    "best_epoch": best_epoch,
                    "budget": budget,
                    "init": init,
                    "optimizer": opt,
                    "scheduler": sched,
                    "test_accuracy": test_acc,
                    "test_precision": test_prec,
                    "test_recall": test_rec,
                    "test_f1": test_f1,
                    "train_loss_curve": epoch_losses,
                    "val_loss_curve": val_losses,
                    "gap_loss_curve": gap_losses,
                },
                ckpt_path
            )

            print(f"Saved checkpoint → {ckpt_path}\n")


def main():
    run_experiments()

if __name__ == "__main__":
    main()