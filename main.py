from decision_tree import DecisionTree
from data_process import process_dataset
import os

# Uncomment if feature dictionary is modified
# if os.path.exists('my_dict.json'):
#     os.remove('my_dict.json')
#     print("Previous feature dictionary deleted.")
print("Data processing...")
train_data, train_label, feature_dict_list, continuous_features, category_features = process_dataset("adult/adult.data")
feature_list = [i for i in range(len(train_data[0]))]
print("Creating decision tree...")
decision_tree = DecisionTree(train_data, train_label, feature_dict_list, continuous_features, threshold=5)
decision_tree.create_tree(train_data, train_label, feature_idx_list=feature_list)
decision_tree.print_tree()
print("Decision tree is created. Output file: decision_tree.txt")
# test
test_data, test_label, test_dict, test_continuous_features, test_category_features = process_dataset("adult/adult.test")
cnt = 0
sum = len(test_data)
for i in range(sum):
    if decision_tree.classify(test_data[i]) == test_label[i]:
        cnt += 1
print("The accuracy of testing dataset is: ")
print(round(cnt / sum, 4))
