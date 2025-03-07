import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



# create testset dataloader
test_data_dir = '/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/embeds_intermediate_answers/segment_by_steps/embeds_intermediate_answers_new/'
# Get all .pt file paths
tests_files = sorted([os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.pt')])

# get the indices of the chunk which contains the full CoT
all_id_list = []
for fp in tests_files:
    data = torch.load(fp)
    id_list = [b['id'] for batch in data['all_batch_info'] for b in batch]
    all_id_list.extend(id_list)

# get the indices of the chunk which contains the full CoT
indices = []
if all_id_list:
    prev_char = all_id_list[0]
    for i in range(1, len(all_id_list)):
        if all_id_list[i] != prev_char:
            indices.append(i-1)
            prev_char = all_id_list[i]
    indices.append(len(all_id_list) - 1)
print(indices) # len: 491



# get the indices of the chunk which contains the full CoT
all_info = [] # len: 3380
for fp in tests_files:
    data = torch.load(fp)
    all_info.extend([b for batch in data['all_batch_info'] for b in batch])

# get the indices of the chunk which contains the full CoT
indices = []
for i in range(len(all_info)):
    if int(all_info[i]['interm_pos']) == all_info[i]['last_one']:
        indices.append(i)
print(indices) # len: 448


# indices = torch.load('/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/testset_res/indices_pos_equal_last.pt')
indices = torch.load('/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/testset_res/indices_id_last.pt')

def get_metrics(val_labels, val_preds):
    # Calculate Metrics
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds)
    return accuracy, precision, recall, f1

################ analysis ##################
res_profile = torch.load('/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/testset_res/res-best_model_weightedloss_e200-hs0-bs64-lr1e-05-wd0.001-alpha2.0-thres0.5-s42-newposw.pt')

# only consider the chunks which contain the full CoT
test_preds = res_profile['test']['test_preds'][indices] # len:491, 448
test_labels = res_profile['test']['test_labels'][indices] # len: 491
test_probs = res_profile['test']['test_probs'][indices] # len: 491

# models's accuracy on answering the questions
acc=test_labels.sum() / 500 #len(test_labels)
print(acc)

# the predictor's accuracy on correctly predicting the final answer's correctness (when only consider the chunks which contain the full CoT)
t_accuracy, t_precision, t_recall, t_f1 = get_metrics(test_labels, test_preds)
print(f"=======Test set========")
print(f"Accuracy: {t_accuracy:.4f}, Precision: {t_precision:.4f}, Recall: {t_recall:.4f}, F1: {t_f1:.4f}")



################ chunks probability within one CoT ##################
# correctly predicted questions
correct_indices, incorrect_indices = [], []
for i in range(len(indices)):
    if all_info[indices[i]]['correctness'] == True:
        correct_indices.append(i)
    else:
        incorrect_indices.append(i)
correct_questions_idx = np.array(indices)[correct_indices] # 443
incorrect_questions_idx = np.array(indices)[incorrect_indices] # 48
# # among the correctly predicted questions (424/500=0.848)
# last_chunk_idx = correct_questions_idx
# first_chunk_idx = [0]
# for i in range(len(correct_questions_idx)-1):
#     first_chunk_idx.append(correct_questions_idx[i]+1)
# first_chunk_idx = np.array(first_chunk_idx)

# among the correctly predicted questions (443/500=0.886)
first_chunk_idx, last_chunk_idx = [0], []
for i in correct_questions_idx:
    id = all_info[i]['id']
    last_chunk_idx.append(i)
    for j in range(i, 0, -1):
        if all_info[j]['id'] != id:
            first_chunk_idx.append(j+1)
            break

# # among them, separate the final answer should be True | False
# final_true_idx, final_false_idx = [], []
# for j in range(len(last_chunk_idx)):
#     if all_info[last_chunk_idx[j]]['correctness'] == True:
#         final_true_idx.append(j)
#     else:
#         final_false_idx.append(j)
# take the preds/probs of chunks
test_preds = res_profile['test']['test_preds'] # len:3380
test_labels = res_profile['test']['test_labels'] # len: 3380
test_probs = res_profile['test']['test_probs']# len: 3380
# predictor's true-probability of first/last chunk
first_chunk_prob, last_chunk_prob = [], []
for k in range(len(last_chunk_idx)):
    correctness = all_info[last_chunk_idx[k]]['correctness']
    # get the true-probability
    last_chunk_prob.append(test_probs[last_chunk_idx[k]])
    first_chunk_prob.append(test_probs[first_chunk_idx[k]])

###### What is the accuracy if we only take the first-chunk?
llabels = test_labels[first_chunk_idx]
ppreds = [1.0 if i > 0.5 else 0.0 for i in first_chunk_prob]
laccuracy, lprecision, lrecall, lf1 = get_metrics(llabels, ppreds)
print(f"=======Test set========")
print(f"Accuracy: {laccuracy:.4f}, Precision: {lprecision:.4f}, Recall: {lrecall:.4f}, F1: {lf1:.4f}")

ppreds = [1.0 if i > 0.5 else 0.0 for i in first_chunk_prob]
llabels_with_gt_answers = test_labels[last_chunk_idx]
laccuracy, lprecision, lrecall, lf1 = get_metrics(llabels_with_gt_answers, ppreds)
print(f"=======Test set========")
print(f"Accuracy: {laccuracy:.4f}, Precision: {lprecision:.4f}, Recall: {lrecall:.4f}, F1: {lf1:.4f}")


# ##### seems some examples, although first-chunk may not enough, but second-chunk is enough
# second_chunk_idx = [i+1 for i in first_chunk_idx] # WRONG!!

# ppreds = [1.0 if i > 0.5 else 0.0 for i in second_chunk_prob]
# llabels_with_gt_answers = test_labels[last_chunk_idx]
# laccuracy, lprecision, lrecall, lf1 = get_metrics(llabels_with_gt_answers, ppreds)
# print(f"=======Test set========")
# print(f"Accuracy: {laccuracy:.4f}, Precision: {lprecision:.4f}, Recall: {lrecall:.4f}, F1: {lf1:.4f}")

# # predictor's true-probability of first/last/second chunk
# first_chunk_prob, last_chunk_prob, second_chunk_prob = [], [], []
# for k in range(len(last_chunk_idx)):
#     correctness = all_info[last_chunk_idx[k]]['correctness']
#     # get the true-probability
#     last_chunk_prob.append(test_probs[last_chunk_idx[k]])
#     first_chunk_prob.append(test_probs[first_chunk_idx[k]])
#     second_chunk_prob.append(test_probs[second_chunk_idx[k]])



import matplotlib.pyplot as plt
import numpy as np

# Replace these with your probability lists
list1 = first_chunk_prob
list2 = last_chunk_prob
# list3 = second_chunk_prob

# Sort the data and compute CDF
sorted_list1 = np.sort(list1)
y1 = np.arange(1, len(sorted_list1) + 1) / len(sorted_list1)
sorted_list2 = np.sort(list2)
y2 = np.arange(1, len(sorted_list2) + 1) / len(sorted_list2)
# sorted_list3 = np.sort(list3)
# y3 = np.arange(1, len(sorted_list3) + 1) / len(sorted_list3)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(sorted_list1, y1, label='first_chunk_true_prob', color='blue')
plt.plot(sorted_list2, y2, label='last_chunk_true_prob', color='red')
# plt.plot(sorted_list3, y3, label='second_chunk_true_prob', color='green')
plt.xlabel("Predictor's true-label confidence")
plt.ylabel('Cumulative Probability')
plt.title('CDF Comparison')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(f'/scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/profile/testset_res/first_last_chunk_true_prob-1.png')








############# for incorrect_questions_idx #############
# among the incorrectly predicted questions (443/500=0.886)
inc_first_chunk_idx, inc_last_chunk_idx = [], []
for i in incorrect_questions_idx:
    id = all_info[i]['id']
    inc_last_chunk_idx.append(i)
    for j in range(i, 0, -1):
        if all_info[j]['id'] != id:
            inc_first_chunk_idx.append(j+1)
            break
#### among the 48 incorrect-examples
# how many of them ever reach to a correct answer?
num = 0
first_time_reach_correct = []
find_the_first_correct = []
for i in range(len(inc_first_chunk_idx)):
    labels = test_labels[inc_first_chunk_idx[i]:inc_last_chunk_idx[i]+1]
    preds = test_preds[inc_first_chunk_idx[i]:inc_last_chunk_idx[i]+1]
    for j in range(len(labels)):
        if labels[j]==1.0:
            num+=1
            first_time_reach_correct.append(j)
            if preds[j]==1.0:
                find_the_first_correct.append(True)
            else:
                find_the_first_correct.append(False)
            break
        else:
            first_time_reach_correct.append(None)
            find_the_first_correct.append(None)

fin_first_time_reach_correct = [i for i in first_time_reach_correct if i is not None]
fin_find_the_first_correct = [i for i in find_the_first_correct if i is not None]




num = 0
inter_label_correct = []
predict_out = []
for i in range(len(inc_first_chunk_idx)):
    labels = test_labels[inc_first_chunk_idx[i]:inc_last_chunk_idx[i]+1]
    preds = test_preds[inc_first_chunk_idx[i]:inc_last_chunk_idx[i]+1]
    if sum(labels)==0:
        inter_label_correct.append(False)
        predict_out.append(None)
        continue
    else:
        inter_label_correct.append(True)
        flag = False
        for j in range(len(labels)):
            if labels[j]==1.0 and preds[j]==1.0:
                predict_out.append(j)
                flag=True
                break
        if flag==False:
            predict_out.append(None)

fin_predict_out = [i for i in predict_out if i is not None]

