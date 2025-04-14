import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from sklearn.model_selection import train_test_split
# from torch.utils.data import RandomSampler
import random


# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class OptimizedBinaryClassificationDataset(IterableDataset):
    def __init__(self, file_list, test_flag):
        self.file_list = file_list.copy()
        self._epoch = 0 # Track epochs
        self.test_flag = test_flag

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            print(f"Worker {worker_info.id} of {worker_info.num_workers}")
        else:
            print("No workers (main process)")

        # Add epoch-based seed variation
        if self.test_flag==False:
            random.seed(42 + self._epoch + (worker_info.id if worker_info else 0))
        
        if worker_info:
            per_worker = len(self.file_list) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.file_list)
            files = self.file_list[start:end]
        else:
            files = self.file_list

        if self.test_flag==False:
            random.shuffle(files)
        # don't shuffle if is for the test dataset
        for fp in files:
            # print(fp)
            try:
                data = torch.load(fp)
                # Process one sample at a time
                # inputs = torch.cat(data['all_last_token_embedding'], dim=0)
                inputs = data['all_last_token_embedding']
                labels = torch.tensor(
                    [b['correctness'] for batch in data['all_batch_info'] for b in batch],
                    dtype=torch.float
                )
                # Yield individual samples without storing full file
                for i in range(inputs.size(0)):
                    yield inputs[i], labels[i]
                del data, inputs, labels # Explicit memory cleanup
                # gc.collect() # Force garbage collection
            except Exception as e:
                print(f"Error in {fp}: {str(e)}")
                continue

    def set_epoch(self, epoch):
        self._epoch = epoch

# get train/val data 
def get_train_val(train_data_dir):
    data_dir = train_data_dir
    # Get all .pt file paths
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')])
    # Split file paths into Train and Validation sets (e.g., 80% Train, 20% Validation)
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    # print("11")
    return train_files, val_files


def get_the_weighted(train_files):
    correctness_list = []
    for fp in train_files:
        data = torch.load(fp)
        labels = torch.tensor(
            [b['correctness'] for batch in data['all_batch_info'] for b in batch], dtype=torch.float
        )
        correctness_list.append(labels)
    correctness_tensor = torch.cat(correctness_list, dim=0)
    positive_count = (correctness_tensor == 1).sum().item() #11739 #15024 
    negative_count = (correctness_tensor == 0).sum().item() #3479 #4670 
    pos_weight = torch.tensor([negative_count / positive_count]) # 0.2964 # 0.3108
    return correctness_tensor, pos_weight


def get_train_loader(train_files):
    # Create dataset and dataloader with optimizations
    train_dataset = OptimizedBinaryClassificationDataset(train_files, test_flag=False)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        num_workers=8,          # Match this to your CPU core count
        prefetch_factor=2,      # Prefetch 2 batches per worker
        pin_memory=True,        # Faster transfer to GPU
        persistent_workers=True # Keep workers alive between epochs
    )
    # print("22")
    return train_dataset, train_loader


def get_val_loader(val_files):
    val_dataset = OptimizedBinaryClassificationDataset(val_files, test_flag=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        num_workers=4,          # Match this to your CPU core count
        prefetch_factor=2,      # Prefetch 2 batches per worker
        pin_memory=True,        # Faster transfer to GPU
        persistent_workers=True # Keep workers alive between epochs
    )
    # print("33")
    return val_loader


# get test data 
def get_test_loader(test_data_dir):
    # Get all .pt file paths
    tests_files = sorted([os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.pt')])
    # create test_dataloader
    test_dataset = OptimizedBinaryClassificationDataset(tests_files, test_flag=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=1,          # Match this to your CPU core count
        prefetch_factor=1,      # Prefetch 2 batches per worker
        pin_memory=True,        # Faster transfer to GPU
        persistent_workers=True # Keep workers alive between epochs
    )
    # print("44")
    return test_loader
