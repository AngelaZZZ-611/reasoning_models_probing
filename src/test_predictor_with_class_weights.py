import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from .dataloader import *
import numpy as np
import argparse
from sklearn.metrics import brier_score_loss
import sys
sys.path.append("../grid_search/")
from compute_metrics import process_file
import os

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

hs_dict = {
        "DeepSeek-R1-Distill-Qwen-32B": 5120,
        "DeepSeek-R1-Distill-Qwen-1.5B": 1536,
        "DeepSeek-R1-Distill-Qwen-7B": 3584,
        "DeepSeek-R1-Distill-Llama-8B": 4096,
        "DeepSeek-R1-Distill-Llama-70B": 8192,
        "QwQ-32B": 5120
    }

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


def get_metrics(val_labels, val_preds):
    # Calculate Metrics
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds)
    macro_f1 = f1_score(val_labels, val_preds, average='macro')
    return accuracy, precision, recall, f1, macro_f1

def load_model(input_size, hidden_size, output_size, ckpt_weight=None):
    if hidden_size==0:
        model = Linear_Model(input_size, output_size)
    else:
        model = MLP(input_size, hidden_size, output_size)
    if ckpt_weight is not None:
        model.load_state_dict(ckpt_weight)
    return model

def load_loss_func(pos_weight, alpha_imbalance_penalty):
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor(pos_weight)
    pos_weight = alpha_imbalance_penalty * pos_weight
    print(f"Alpha * Positive Class Weight: {pos_weight.item():.4f}")
    # Define loss function with calculated pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    return criterion

def calculate_ece(y_true, y_pred, n_bins=10):
    """
    Calculate the Estimated Calibration Error (ECE).

    Parameters:
    y_true (np.array): True labels (0 or 1).
    y_pred (np.array): Predicted probabilities.
    n_bins (int): Number of bins to use for calibration.

    Returns:
    float: The Estimated Calibration Error.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Bin the predictions
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine which samples fall into this bin
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        if np.sum(in_bin) == 0:
            continue  # Skip empty bins
        # Calculate accuracy and confidence in this bin
        bin_acc = np.mean(y_true[in_bin])
        bin_conf = np.mean(y_pred[in_bin])
        # Update ECE
        ece += np.abs(bin_acc - bin_conf) * np.sum(in_bin)
    # Normalize by the total number of samples
    ece /= len(y_true)
    return ece

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default=None, help='train data path.') # train-set
    parser.add_argument('--test_data_dir', type=str, default=None, help='test data path.') # test-set
    # parser.add_argument('--epochs', type=int, default=200,
    #                     help='number of the uper-bound epochs to train the model ')
    parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--lr', type=float, default=1e-5)
    # parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=4096)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha_imbalance_penalty', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--checkpoint_model_path', type=str, default=None, help='path for saving the best model. ') 
    parser.add_argument('--grid_search_result_path', type=str, default=None, help='path for saving grid search results, can be used for automatically deciding the best model. ')
    parser.add_argument('--metric', type=str, default='best_val_acc', help='metrics for helping decide the best model. ')
    parser.add_argument('--topk', type=int, default=1, help='howmany models to eval')
    parser.add_argument('--pos_weight', type=float, default=None, help='train pos weight')

    parser.add_argument('--save_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/testset_res', help='path for saving profile for testset results. ') 
    parser.add_argument('--overwrite', action="store_true", help='overwrite the existing result file. ')
    parser.add_argument('--model_name', type=str, default=None, help='path for saving the best model. ') 
    args = parser.parse_args()
    print(args)

    # if args.overwrite:
    #     if os.path.exists(args.save_path):
    #         os.system(f"rm -rf {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)

    # Set all seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = args.pos_weight
        print(f"Positive Class Weight: {pos_weight}")
    if args.train_data_dir:
        # load data
        train_files, val_files = get_train_val(args.train_data_dir)

        # Calculate Class Weights
        _, pos_weight = get_the_weighted(train_files)
        print(f"Positive Class Weight from training data: {pos_weight.item():.4f}") # 0.2964 # 0.3108

    

    # Model parameters
    # input_size = args.input_size #X.shape[1]
    
    input_size = hs_dict[args.model_name]
    hidden_size = args.hidden_size #0/16/32/1024
    output_size = args.output_size  # Binary output
    


    
    # learning_rate = args.lr #1e-4 #0.001
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.wd) # Start with a moderate LR

    # load checkpoint model (e.g., the best model)
    if args.checkpoint_model_path:
        if args.checkpoint_model_path[-3:] == '.pt':
            checkpoint_path = args.checkpoint_model_path
        else:
            checkpoint_path = f"{args.checkpoint_model_path}/best_model_weightedloss_e200-hs{args.hidden_size}-bs64-lr{args.lr}-wd{args.wd}-alpha{args.alpha_imbalance_penalty}-thres0.5-s42.pt"
    else:
        assert args.grid_search_result_path is not None, "Please provide the path for the grid search results when ckpt path is not explicitly specified"
        import json
        import pandas as pd
        res_path = args.grid_search_result_path
        with open(res_path, "r") as f:
            res = [json.loads(line) for line in f]
        df = pd.DataFrame(res)
        df = df.sort_values(args.metric, ascending=False)
        df = df.reset_index(drop=True)
        checkpoint_path = list(df['best_ckpt'][:args.topk].values)
        print(f"Top {args.topk} models: {checkpoint_path}")

    if isinstance(checkpoint_path, str):
        checkpoint_path = [checkpoint_path]

    for ckpt in checkpoint_path:
        fname = ckpt.split('/')[-1]
        test_data_name = args.test_data_dir.split('/')[-1]
        output_file = f'{args.save_path}/res-{test_data_name}-{fname}'
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping...")
            continue

        print(f"Loading checkpoint from {ckpt}")
        base_ckpt_name = ckpt.split('/')[-1]
        hidden_size = int(base_ckpt_name.split('-')[1][2:])
        ckpt_weights = torch.load(ckpt)
        
        if "pos_weight_from_train" in ckpt_weights:
            if pos_weight is not None:
                print(f"Overwriting the pos_weight from the checkpoint with the one from the ckpt")
            pos_weight = ckpt_weights["pos_weight_from_train"]
            print(f"Positive Class Weight: {pos_weight}")
            ckpt_weights = ckpt_weights["model"]
        # assert pos_weight is not None, "Please provide the positive class weight for the model"
        model = load_model(input_size, hidden_size, output_size, ckpt_weights)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(device)

        alpha_imbalance_penalty = base_ckpt_name.split('-')[-3][5:]
        if alpha_imbalance_penalty == "None":
            alpha_imbalance_penalty = None
        else:
            alpha_imbalance_penalty = float(alpha_imbalance_penalty)
        criterion = load_loss_func(pos_weight, alpha_imbalance_penalty)

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
        ece = calculate_ece(test_labels, test_probs)
        brier_loss = brier_score_loss(test_labels, test_probs)
        t_accuracy, t_precision, t_recall, t_f1, t_macrof1 = get_metrics(test_labels, test_preds)
        profile['test']['metrics'] = {'accuracy': t_accuracy, 'precision': t_precision, 'recall': t_recall, 'f1': t_f1, "macro_f1": t_macrof1, "ece": ece, "brier_loss": brier_loss}
        print(f"=======Test set========")
        print(f"Accuracy: {t_accuracy:.4f}, Precision: {t_precision:.4f}, Recall: {t_recall:.4f}, F1: {t_f1:.4f}, macro F1: {t_macrof1:.4f}; ece: {ece:.4f}, brier_loss: {brier_loss:.4f}; loss: {test_loss:.4f}")



        # Save the profile dictionary
        fname = ckpt.split('/')[-1]
        test_data_name = args.test_data_dir.split('/')[-1]
        profile_path = f'{args.save_path}/res-{test_data_name}-{fname}'
        torch.save(profile, profile_path)

        metrics = process_file(profile_path)

        # save result in json
        import json
        def default_converter(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, complex):
                return [obj.real, obj.imag]
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        with open(f'{args.save_path}/test_result.json', 'a+') as f:
            f.writelines(
                json.dumps(
                    {
                        'checkpoint_path': ckpt,
                        "train_data": args.train_data_dir,
                        "test_data": args.test_data_dir,
                        **metrics,
                    }, default=default_converter
                ) + "\n"
            )


if __name__ == '__main__':
    main()

#python test_predictor_with_class_weights.py --lr 1e-05 --wd 0.001 --hidden_size 0 --alpha_imbalance_penalty 2.0 
