import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataloader import *
import numpy as np
import argparse
from probe_model import load_model,  hs_dict

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def run_eval(args, epoch, model, criterion, val_loader):
    print(f'======validating epoch {epoch+1}======')
    model.eval()
    val_preds, val_labels, val_probs = [], [], []
    val_loss = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move inputs and labels to GPU
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward Pass
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)  # Apply Sigmoid for probabilities
            # preds = (probs > 0.5).float()   # Using 0.5 as threshold
            preds = (probs > args.threshold).float()

            val_probs.extend(probs.cpu().numpy()) # Move to CPU before converting to NumPy
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_loss.append(loss.item())
    val_epoch_loss = np.mean(val_loss)
    return val_labels, val_preds, val_probs, val_epoch_loss

# 1. Add checkpoint saving function
def save_checkpoint(args, epoch, model, optimizer, loss):
    path = args.save_model_path
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    filename = os.path.join(path, f"checkpoint_weightedloss_e{args.epochs}_e{epoch}-hs{args.hidden_size}-bs{args.batch_size}-lr{args.lr}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{args.seed}.pt")
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint at epoch {epoch} to {filename}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default=None, help='train data path.') # train-set
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of the uper-bound epochs to train the model ')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=4096)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha_imbalance_penalty', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_model_path', type=str, default=None, help='path for saving the best model. ') 
    parser.add_argument('--store_path', type=str, default=None, help='path for saving profile. ') 
    parser.add_argument('--model_name', type=str, default=None, help='path for saving the best model. ') 

    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_model_path, exist_ok=True)
    os.makedirs(args.store_path, exist_ok=True)

    # Set all seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load data
    train_files, val_files = get_train_val(args.train_data_dir)

    # Calculate Class Weights
    pos_weight_from_train = None
    if args.alpha_imbalance_penalty is not None:
        _, pos_weight_from_train = get_the_weighted(train_files)
        print(f"Positive Class Weight: {pos_weight_from_train.item():.4f}") # 0.2964 # 0.3108
        pos_weight = args.alpha_imbalance_penalty * pos_weight_from_train
        print(f"Alpha * Positive Class Weight: {pos_weight.item():.4f}") 

    # Model parameters
    # input_size = args.input_size #X.shape[1]
    
    input_size = hs_dict[args.model_name]
    hidden_size = args.hidden_size #0/16/32/1024
    output_size = args.output_size  # Binary output

    model = load_model(input_size, hidden_size, output_size)
    model.to(device)

    # Define loss function with calculated pos_weight
    if args.alpha_imbalance_penalty is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    learning_rate = args.lr #1e-4 #0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.wd) # Start with a moderate LR


    profile = {'args': vars(args), 'train': {'epochs_loss': []}, 'val': {'val_labels': [], 'val_preds': [], 'val_probs': [], 'val_epoch_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}}
    # Track best loss for saving
    patience = 10
    best_val_loss = np.inf
    best_metrics = {}
    epochs_no_improve = 0

    # train_loader, val_loader
    train_dataset, train_loader = get_train_loader(train_files)
    val_loader = get_val_loader(val_files)

    # Training Loop
    epochs = args.epochs #200
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)  # Reset dataset state
        # Training Phase
        model.train()
        running_loss = 0.0  # Track loss for each epoch
        print(f"======training epoch {epoch+1}======")
        for i, (inputs, labels) in enumerate(train_loader):
            # print(f"{i}, {inputs.shape}, {labels.shape}")
            # Move inputs and labels to GPU
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward Pass
            outputs = model(inputs).squeeze(-1)  # No Sigmoid here
            loss = criterion(outputs, labels)
            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Print loss every N iterations
            if i==0 or (i+1) % 100 == 0:  # Change 10 to your preferred frequency
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {loss.item():.4f}")
        
        # Average loss for the epoch
        epoch_loss = running_loss / (i+1)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss:.4f}")
        profile['train']['epochs_loss'].append(epoch_loss)

        # Evaluation Phase
        val_labels, val_preds, val_probs, val_epoch_loss = run_eval(args, epoch, model, criterion, val_loader=val_loader)

        # Convert to NumPy arrays for metrics
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)
        profile['val']['val_labels'].append(val_labels)
        profile['val']['val_preds'].append(val_preds)
        profile['val']['val_probs'].append(val_probs)
        profile['val']['val_epoch_loss'].append(np.array(val_epoch_loss))

        # Calculate Metrics
        accuracy = accuracy_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, zero_division=0)
        recall = recall_score(val_labels, val_preds, zero_division=0)
        f1 = f1_score(val_labels, val_preds)

        # Print Epoch Metrics
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {val_epoch_loss.item():.4f} | Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
        print()
        profile['val']['accuracy'].append(accuracy)
        profile['val']['precision'].append(precision)
        profile['val']['recall'].append(recall)
        profile['val']['f1'].append(f1)

        # # save a checkpoint model for each 100 epochs training
        # if (epoch + 1) % 100 == 0:
        #     save_checkpoint(args, epoch + 1, model, optimizer, epoch_loss)
        
        # Early stopping logic
        best_model_path = os.path.join(f"{args.save_model_path}", f"best_model_weightedloss_e{args.epochs}-hs{args.hidden_size}-bs{args.batch_size}-lr{args.lr}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{args.seed}.pt")
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
            epochs_no_improve = 0
            # Save the best model
            torch.save({"model": model.state_dict(), "pos_weight_from_train": pos_weight_from_train}, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch}')
                break


    # save final model
    final_model_path = f'{args.save_model_path}/fin-model-weightedloss-e{epochs}-hs{hidden_size}-bs{args.batch_size}-lr{learning_rate}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{seed}.pt'
    torch.save({"model": model.state_dict(), "pos_weight_from_train": pos_weight_from_train}, final_model_path)

    # save profile
    torch.save(profile, f'{args.store_path}/profile-weightedloss-e{epochs}-hs{hidden_size}-bs{args.batch_size}-lr{learning_rate}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{seed}.pt')
    # save profile as json
    import json, datetime
    def default_converter(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, complex):
            return [obj.real, obj.imag]
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    grid_search_res_path = "/".join(args.save_model_path.split("/")[:-1])
    with open(os.path.join(grid_search_res_path, "grid_search_result.jsonl"), "a+") as f:
        f.write(json.dumps(
            {
                "lr": learning_rate,
                "wd": args.wd,
                "hidden_size": hidden_size,
                "alpha_imbalance_penalty": args.alpha_imbalance_penalty,
                "best_val_acc": best_metrics['accuracy'],
                "best_val_precision": best_metrics['precision'],
                "best_val_recall": best_metrics['recall'],
                "best_val_f1": best_metrics['f1'],
                "best_eval_loss": best_val_loss,
                "best_ckpt": best_model_path if os.path.exists(best_model_path) else final_model_path,
            }, default=default_converter) + "\n")
    with open(f'{args.store_path}/profile-weightedloss-e{epochs}-hs{hidden_size}-bs{args.batch_size}-lr{learning_rate}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{seed}.json', 'w') as f:
        json.dump(profile, f, default=default_converter)


if __name__ == '__main__':
    main()
