import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataloader import *
import numpy as np
import argparse


# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# In your MLP Model, remove Sigmoid activation from the output layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)  # No Sigmoid here
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)  # Raw logits
        return x

class Linear_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear_Model, self).__init__()
        self.output = nn.Linear(input_size, output_size)  # No Sigmoid here

    def forward(self, x):
        x = self.output(x)  # Raw logits
        return x

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
    filename = os.path.join(path, f"checkpoint_weightedloss_e{args.epochs}_e{epoch}-hs{args.hidden_size}-bs64-lr{args.lr}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{args.seed}.pt")
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint at epoch {epoch} to {filename}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/embeds_intermediate_answers/train_dataset_MATH', help='train data path.') # train-set
    parser.add_argument('--test_data_dir', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/embeds_intermediate_answers/test_dataset_MATH', help='test data path.') # train-set
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of the uper-bound epochs to train the model ')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=4096)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha_imbalance_penalty', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_model_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/grid_search/checkpoints', help='path for saving the best model. ') 
    parser.add_argument('--store_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/grid_search/store', help='path for saving profile. ') 
    args = parser.parse_args()
    print(args)

    # Set all seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load data
    train_files, val_files = get_train_val(args.train_data_dir)

    # Calculate Class Weights
    _, pos_weight = get_the_weighted(train_files)
    print(f"Positive Class Weight: {pos_weight.item():.4f}") # 0.2964 # 0.3108
    pos_weight = args.alpha_imbalance_penalty * pos_weight
    print(f"Alpha * Positive Class Weight: {pos_weight.item():.4f}") 

    # Model parameters
    input_size = args.input_size #X.shape[1]
    hidden_size = args.hidden_size #0/16/32/1024
    output_size = args.output_size  # Binary output
    if args.hidden_size==0:
        model = Linear_Model(input_size, output_size)
    else:
        model = MLP(input_size, hidden_size, output_size)
    model.to(device)

    # Define loss function with calculated pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    learning_rate = args.lr #1e-4 #0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.wd) # Start with a moderate LR


    profile = {'args': args, 'train': {'epochs_loss': []}, 'val': {'val_labels': [], 'val_preds': [], 'val_probs': [], 'val_epoch_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}}
    # Track best loss for saving
    patience = 10
    best_val_loss = np.inf
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
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(f"{args.save_model_path}", f"best_model_weightedloss_e{args.epochs}-hs{args.hidden_size}-bs64-lr{args.lr}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{args.seed}.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch}')
                break


    # save final model
    torch.save(model.state_dict(), f'{args.save_model_path}/fin-model-weightedloss-e{epochs}-hs{hidden_size}-bs64-lr{learning_rate}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{seed}.pt')

    # save profile
    torch.save(profile, f'{args.store_path}/profile-weightedloss-e{epochs}-hs{hidden_size}-bs64-lr{learning_rate}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres{args.threshold}-s{seed}.pt')


if __name__ == '__main__':
    main()
