from tree_node import TreeNode
from data_process import process_dataset

class DecisionTree:
    def __init__(self, train_data, train_label, feature_dict, root=None, threshold=5):
        '''
        :param train_data:
        :param train_label:
        :param feature_dict:
        :param threshold: when |S| < threshold, it is too small
        '''
        self.train_data = train_data
        self.train_label = train_label
        self.feature_dict = feature_dict
        self.root = None
        self.threshold = threshold

    def check_data(self, data_list, label_list):
        # check data format
        # length of train_data must be the same with train_label
        if len(data_list) != len(label_list):
            return False
        # length of feature_dict must match train_data & train_label
        if len(data_list[0])+1 != len(self.feature_dict):
            return False

        return True

    def create_tree(self, data_list, label_list, attr_idx=-1, attr_val=-1):
        '''
        :param data_list: data-vector list
        :param label_list: label list of data_list
        :param attr_idx: attribute index of the split
        :param attr_val: encode attribute of the split
        :return: TreeNode
        '''
        if not self.check_data(data_list, label_list):
            print("error in data_list & label_list")
            return None

        print("creating tree... attr_idx=" + str(attr_idx) + " attr_val=" + str(attr_val))
        tree_node = TreeNode(attr_idx=attr_idx, attr_val=attr_val)
        if self.root is None:
            self.root = tree_node

        # if all objects belong to the same class
        only_label = self.is_same_class(label_list)
        if only_label != -1:
            tree_node.is_leave = True
            tree_node.result = only_label
            # print("all objects belong to the same class:" + " only_label=" + str(only_label))
            return tree_node  # return a leaf node with the value of this class

        # if all objects have the same attribute
        # or |S| is too small
        if self.is_same_attribute(data_list) or len(label_list) < self.threshold:
            tree_node.is_leave = True
            tree_node.result = self.get_majority_label(label_list)
            print("all objects have the same attribute")
            return tree_node

        # find the best split
        if attr_idx == -1:
            feature = self.feature_dict[0]
        else:
            feature = self.feature_dict[attr_idx]
        # get feature index for next split
        if attr_idx == len(self.feature_dict) - 1 - 1: # index out of scope
            tree_node.is_leave = True
            tree_node.result = self.get_majority_label(label_list)
            return tree_node
        else:
            next_attr_idx = attr_idx + 1
        # find split with best GINI
        gini_list = []
        key_list = []
        for key in feature.keys():
            data_list1, label_list1, data_list2, label_list2 = self.split_dataset(
                data_list, label_list, next_attr_idx, key)
            gini = self.cal_split_gini(data_list1, label_list1, data_list2, label_list2)
            gini_list.append(gini)
            key_list.append(key)
        best_index = gini_list.index(min(gini_list))  # find min gini
        best_attr = key_list[best_index]  # encode attr
        # split into subset S1 and S2
        data_list1, label_list1, data_list2, label_list2 = self.split_dataset(
            data_list, label_list, next_attr_idx, best_attr)
        # create tree for S1 and S2
        if len(data_list1) > 0:
            tree_node.true_brunch = self.create_tree(data_list1, label_list1, next_attr_idx, best_attr)
        else: # s1 is empty
            result = self.get_majority_label(label_list)
            tree_node.true_brunch = TreeNode(is_leave=True, result=result, attr_idx=next_attr_idx, attr_val=best_attr)
        if len(data_list2) > 0:
            tree_node.false_brunch = self.create_tree(data_list2, label_list2, next_attr_idx, best_attr)
        else: # s2 is empty
            result = self.get_majority_label(label_list)
            tree_node.false_brunch = TreeNode(is_leave=True, result=result, attr_idx=next_attr_idx, attr_val=best_attr)

        return tree_node

    def is_same_class(self, label_list):
        first_label = label_list[0]
        for label in label_list:
            if first_label != label:
                return -1
        return first_label

    def is_same_attribute(self, data_list):
        first_row = data_list[0]
        for row in data_list:
            if not row == first_row:
                return False
        return True

    @staticmethod
    def get_majority_label(label_list):
        return max(label_list, key=label_list.count)

    @staticmethod
    def split_dataset(data_list, label_list, attr_idx, attr_val):
        data_list1 = []
        data_list2 = []
        label_list1 = []
        label_list2 = []
        for data, label in zip(data_list, label_list):
            if data[attr_idx] <= attr_val:
                data_list1.append(data)
                label_list1.append(label)
            else:
                data_list2.append(data)
                label_list2.append(label)

        return data_list1, label_list1, data_list2, label_list2

    def cal_split_gini(self, data_list1, label_list1, data_list2, label_list2):
        s1 = len(data_list1)
        s2 = len(data_list2)
        s = s1 + s2
        gini1 = self.cal_gini(data_list1, label_list1)
        gini2 = self.cal_gini(data_list2, label_list2)
        return round(s1/s, 2) * gini1 + round(s2/s, 2) * gini2

    @staticmethod
    def cal_gini(data_list, label_list):
        n = len(data_list)
        if n == 0:
            return 1
        ny = len(list(filter(lambda x: x == 1, label_list)))
        py = round(ny/n, 2)
        pn = 1 - py
        return 1 - (py * py + pn * pn)

    def classify(self, data):
        return self.traverse(self.root, data)

    def traverse(self, root, data):
        if root is None:
            return -1
        if root.is_leave:
            return root.result
        if root.true_brunch:
            attr_idx = root.true_brunch.attr_idx
            attr_val = root.true_brunch.attr_val
            if data[attr_idx] == attr_val:
                return self.traverse(root.true_brunch, data)
        if root.false_brunch:
            return self.traverse(root.false_brunch, data)
        return -1



if __name__ == '__main__':
    train_data, train_label, feature_dict = process_dataset()
    decision_tree = DecisionTree(train_data,train_label,feature_dict, threshold=5)
    decision_tree.create_tree(train_data,train_label)
    # test
    test_data, test_label, test_dict = process_dataset("./adult/adult.test")
    cnt = 0
    sum = len(test_data)
    for i in range(sum):
        if decision_tree.classify(test_data[i]) == test_label[i]:
            cnt += 1

    print(round(cnt / sum, 4))

