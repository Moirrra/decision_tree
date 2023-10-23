import math

from tree_node import TreeNode
from data_process import process_dataset
from itertools import combinations

class DecisionTree:
    def __init__(self, train_data, train_label, feature_dict_list, continuous_features, root=None, threshold=5):
        '''
        :param train_data: encode attribute list
        :param train_label: encode label list
        :param feature_dict_list: dict list  key:encode val:feature (attribute & label)
        :param threshold: when |S| < threshold, it is too small
        '''
        self.train_data = train_data
        self.train_label = train_label
        self.feature_dict_list = feature_dict_list
        self.continuous_features = continuous_features
        self.root = None
        self.threshold = threshold

    def check_data(self, data_list, label_list):
        # check data format
        # length of train_data must be the same with train_label
        if len(data_list) != len(label_list):
            return False
        # length of feature_dict must match train_data & train_label
        if len(data_list[0])+1 != len(self.feature_dict_list):
            return False

        return True

    def create_tree(self, data_list, label_list, attr_idx=-1, attr_val=-1, feature_idx_list=None):
        '''
        :param data_list: data-vector list
        :param label_list: label list of data_list
        :param attr_idx: attribute index of the split
        :param attr_val: encode attribute of the split
        :param feature_idx_list: list of feature indexes not split
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
        # get feature index for next split
        if len(feature_idx_list) == 0:
            tree_node.is_leave = True
            tree_node.result = self.get_majority_label(label_list)
            return tree_node
        # find split with best GINI
        best_attr_idx, best_attr_val = self.find_best_split(data_list, label_list, feature_idx_list)
        feature_idx_list.remove(best_attr_idx)
        # split into subset S1 and S2
        if best_attr_idx in self.continuous_features:
            data_list1, label_list1, data_list2, label_list2 = self.split_dataset_continuous(
                data_list, label_list, best_attr_idx, best_attr_val)
        else:
            data_list1, label_list1, data_list2, label_list2 = self.split_dataset_category2(
                data_list, label_list, best_attr_idx, best_attr_val)
        print("s1:" + str(len(data_list1)) + " s2:" + str(len(data_list2)))
        # create tree for S1 and S2
        if len(data_list1) > 0:
            tree_node.true_brunch = self.create_tree(data_list1, label_list1, best_attr_idx, best_attr_val, feature_idx_list)
        else:  # s1 is empty
            result = self.get_majority_label(label_list)
            tree_node.true_brunch = TreeNode(is_leave=True, result=result, attr_idx=best_attr_idx, attr_val=best_attr_val)
        if len(data_list2) > 0:
            tree_node.false_brunch = self.create_tree(data_list2, label_list2, best_attr_idx, -1, feature_idx_list)
        else:  # s2 is empty
            result = self.get_majority_label(label_list)
            tree_node.false_brunch = TreeNode(is_leave=True, result=result, attr_idx=best_attr_idx, attr_val=-1)

        return tree_node

    def find_best_split(self, data_list, label_list, feature_idx_list):
        best_gini = 0.5
        best_feature_idx = -1
        best_feature_val = -1
        for feature_idx in feature_idx_list:
            feature_dict = self.feature_dict_list[feature_idx]
            min_gini = 0.5
            feature_val = -1
            if feature_idx in self.continuous_features:  # continuous features
                for val in feature_dict.keys():
                    data_list1, label_list1, data_list2, label_list2 = self.split_dataset_continuous(
                        data_list, label_list, feature_idx, val)
                    # print(str(len(data_list1)) + " " + str(len(data_list2)))
                    gini = self.cal_split_gini(data_list1, label_list1, data_list2, label_list2)
                    if gini < min_gini:
                        min_gini = gini
                        feature_val = val
            else:  # category features
                for val in feature_dict.keys():
                    data_list1, label_list1, data_list2, label_list2 = self.split_dataset_category2(
                        data_list, label_list, feature_idx, val)
                    # print(str(len(data_list1)) + " " + str(len(data_list2)))
                    gini = self.cal_split_gini(data_list1, label_list1, data_list2, label_list2)
                    if gini < min_gini:
                        min_gini = gini
                        feature_val = val
                # val_cnt = len(feature_dict)
                # idx_list = [i for i in range(val_cnt)]
                # condition_list = []
                # for i in range(math.floor(len(idx_list)/2)):
                #     condition_list += list(combinations(idx_list,i+1))
                # condition_list = [list(x) for x in condition_list]
                # for condition in condition_list:
                #     data_list1, label_list1, data_list2, label_list2 = self.split_dataset_category(
                #         data_list, label_list, feature_idx, condition)
                #     gini = self.cal_split_gini(data_list1, label_list1, data_list2, label_list2)
                #     if gini < min_gini:
                #         min_gini = gini
                #         feature_val = condition
            if min_gini < best_gini:
                best_gini = min_gini
                best_feature_idx = feature_idx
                best_feature_val = feature_val
        print("best gini = " + str(best_gini) + " best_feature_idx = " + str(best_feature_idx) +
              " best_feature_val = " + str(best_feature_val))
        return best_feature_idx, best_feature_val

    @staticmethod
    def is_same_class(label_list):
        first_label = label_list[0]
        for label in label_list:
            if first_label != label:
                return -1
        return first_label

    @staticmethod
    def is_same_attribute(data_list):
        first_row = data_list[0]
        for row in data_list:
            if not row == first_row:
                return False
        return True

    @staticmethod
    def get_majority_label(label_list):
        return max(label_list, key=label_list.count)

    def split_dataset_continuous(self, data_list, label_list, attr_idx, attr_val):
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

    # todo: long time to test every combination of category feature
    def split_dataset_category(self, data_list, label_list, attr_idx, condition):
        data_list1 = []
        data_list2 = []
        label_list1 = []
        label_list2 = []
        condition_dict = dict.fromkeys(condition, 0)
        for data, label in zip(data_list, label_list):
            if condition_dict.get(data[attr_idx]):
            # if data[attr_idx] in condition:
                data_list1.append(data)
                label_list1.append(label)
            else:
                data_list2.append(data)
                label_list2.append(label)
        return data_list1, label_list1, data_list2, label_list2

    # only choose one of category feature value to split
    def split_dataset_category2(self, data_list, label_list, attr_idx, attr_val):
        data_list1 = []
        data_list2 = []
        label_list1 = []
        label_list2 = []
        for data, label in zip(data_list, label_list):
            if data[attr_idx] == attr_val:
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
        return round(s1 / s * gini1 + s2 / s * gini2, 6)

    @staticmethod
    def cal_gini(data_list, label_list):
        n = len(data_list)
        if n == 0:
            return 1
        # ny = len(list(filter(lambda x: x == 1, label_list)))
        ny = 0
        for label in label_list:
            if label == 1:
                ny += 1
        py = ny / n
        pn = 1 - py
        return round(1 - (py * py + pn * pn), 6)

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
    train_data, train_label, feature_dict_list, continuous_features, category_features = process_dataset("adult/adult.data")
    decision_tree = DecisionTree(train_data, train_label, feature_dict_list, continuous_features, threshold=5)
    feature_list = [i for i in range(len(train_data[0]))]
    print(feature_list)
    decision_tree.create_tree(train_data, train_label, feature_idx_list=feature_list)
    # test
    test_data, test_label, test_dict, test_continuous_features, test_category_features = process_dataset("adult/adult.test")
    cnt = 0
    sum = len(test_data)
    for i in range(sum):
        if decision_tree.classify(test_data[i]) == test_label[i]:
            cnt += 1
    print(round(cnt / sum, 4))