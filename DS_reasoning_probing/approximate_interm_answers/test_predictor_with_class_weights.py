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

def run_eval(args, model, criterion, val_loader):
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
            outputs = model(inputs).squeeze()
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


def get_metrics(val_labels, val_preds):
    # Calculate Metrics
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds)
    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/embeds_intermediate_answers/train_dataset_MATH', help='train data path.') # train-set
    parser.add_argument('--test_data_dir', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/embeds_intermediate_answers/test_dataset_MATH', help='test data path.') # test-set
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
    parser.add_argument('--checkpoint_model_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/grid_search/checkpoints', help='path for saving the best model. ') 
    parser.add_argument('--save_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/testset_res', help='path for saving profile for testset results. ') 
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

    # load checkpoint model (e.g., the best model)
    checkpoint_path = f"{args.checkpoint_model_path}/best_model_weightedloss_e200-hs{args.hidden_size}-bs64-lr{args.lr}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres0.5-s42.pt"
    checkpoint = torch.load(checkpoint_path)
    print(f"Loading checkpoint from {checkpoint_path}")

    model.load_state_dict(checkpoint)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)

    profile = {'val': {}, 'test': {}}
    # #### run test on validation dataset
    # val_labels, val_preds, val_probs, val_loss = run_eval(args, model, criterion, val_loader=val_loader)

    # # Convert to NumPy arrays for metrics
    # profile['val']['val_labels'] = np.array(val_labels)
    # profile['val']['val_preds']= np.array(val_preds)
    # profile['val']['val_probs'] = np.array(val_probs)
    # profile['val']['val_loss'] = np.array(val_loss)

    # # Calculate Metrics
    # accuracy, precision, recall, f1 = get_metrics(val_labels, val_preds)
    # profile['val']['metrics'] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    # print(f"=======Validation set========")
    # print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}; loss: {val_loss:.4f}")

    #### run test on the test dataset
    test_loader = get_test_loader(args.test_data_dir)
    test_labels, test_preds, test_probs, test_loss = run_eval(args, model, criterion, val_loader=test_loader)

    # Convert to NumPy arrays for metrics
    profile['test']['test_labels'] = np.array(test_labels)
    profile['test']['test_preds']= np.array(test_preds)
    profile['test']['test_probs'] = np.array(test_probs)
    profile['test']['test_loss'] = np.array(test_loss)

    # Calculate Metrics
    t_accuracy, t_precision, t_recall, t_f1 = get_metrics(test_labels, test_preds)
    profile['test']['metrics'] = {'accuracy': t_accuracy, 'precision': t_precision, 'recall': t_recall, 'f1': t_f1}
    print(f"=======Test set========")
    print(f"Accuracy: {t_accuracy:.4f}, Precision: {t_precision:.4f}, Recall: {t_recall:.4f}, F1: {t_f1:.4f}; loss: {test_loss:.4f}")



    # Save the profile dictionary
    fname = checkpoint_path.split('/')[-1]
    torch.save(profile, f'{args.save_path}/res-{fname}')


if __name__ == '__main__':
    main()

#python test_predictor_with_class_weights.py --lr 1e-05 --wd 0.001 --hidden_size 0 --alpha_imbalance_penalty 2.0 
