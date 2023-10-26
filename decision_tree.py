from tree_node import TreeNode


class DecisionTree:
    """ The definition of DecisionTree

    :param train_data: encode attribute list
    :param train_label: encode label list
    :param feature_dict_list: dict list  key:encode val:feature (attribute & label)
    :param continuous_features: index of continuous feature
    :param root: root treenode
    :param threshold: when |S| < threshold, it is too small
    :type train_data:
    :type train_label: List[int]
    :type feature_dict_list: List[Dict[int,str]]
    :type root: TreeNode
    :type threshold: int
    """
    def __init__(self, train_data, train_label, feature_dict_list, continuous_features, root=None, threshold=5):
        self.train_data = train_data
        self.train_label = train_label
        self.feature_dict_list = feature_dict_list
        self.continuous_features = continuous_features
        self.root = root
        self.threshold = threshold

    def check_data(self, data_list, label_list):
        """ Check data format of data_list & label_list

        :param data_list: subset of training data
        :param label_list: subset of training label
        :type data_list: List[List[int]]
        :type label_list: List[int]
        :raise ValueError
        """
        # length of train_data must be the same with train_label
        if len(data_list) != len(label_list):
            raise ValueError("length of data_list does NOT match length of label_list")
        # length of feature_dict must match train_data & train_label
        if len(data_list[0])+1 != len(self.feature_dict_list):
            raise ValueError("length of data_list does NOT match length of label_list")

    def create_tree(self, data_list, label_list, attr_idx=-1, attr_val=-1, feature_idx_list=None):
        """ Create decision tree recursively

        :param data_list: subset of training data
        :param label_list: subset of training label
        :param attr_idx: attribute index of the split
        :param attr_val: encode attribute value of the split
        :param feature_idx_list: list of feature indexes not split
        :type data_list: List[List[int]]
        :type label_list: [List[int]
        :type attr_idx: int
        :type attr_val: int
        :type feature_idx_list: List[int]
        :return: treenode with subtrees
        :rtype: TreeNode
        """
        try:
            self.check_data(data_list, label_list)
        except Exception as e:
            print(str(e))
            return None

        tree_node = TreeNode(attr_idx=attr_idx, attr_val=attr_val)
        # initialize root node for the first time
        if self.root is None:
            self.root = tree_node

        # if all objects belong to the same class
        only_label = self.is_same_class(label_list)
        if only_label != -1:
            tree_node.is_leaf = True
            tree_node.result = only_label
            return tree_node  # return a leaf node with the value of this class

        # if all objects have the same attribute
        # or |S| is too small
        if self.is_same_attribute(data_list) or len(label_list) < self.threshold:
            tree_node.is_leaf = True
            tree_node.result = self.get_majority_label(label_list)
            return tree_node

        # find the best split
        # get feature index for next split
        if len(feature_idx_list) == 0:
            tree_node.is_leaf = True
            tree_node.result = self.get_majority_label(label_list)
            return tree_node
        # find split with best GINI
        best_attr_idx, best_attr_val = self.find_best_split(data_list, label_list, feature_idx_list)
        feature_idx_list.remove(best_attr_idx)
        # split into subset S1 and S2
        if best_attr_idx in self.continuous_features:
            data_list1, label_list1, data_list2, label_list2 = self.split_dataset(
                data_list, label_list, best_attr_idx, best_attr_val, is_continuous=True)
        else:
            data_list1, label_list1, data_list2, label_list2 = self.split_dataset(
                data_list, label_list, best_attr_idx, best_attr_val, is_continuous=False)

        # create tree for S1 and S2
        if len(data_list1) > 0:  # get subtree
            tree_node.true_brunch = self.create_tree(data_list1, label_list1, best_attr_idx, best_attr_val, feature_idx_list)
        else:  # s1 is empty
            result = self.get_majority_label(label_list)
            tree_node.true_brunch = TreeNode(is_leaf=True, result=result, attr_idx=best_attr_idx, attr_val=best_attr_val)
        if len(data_list2) > 0:  # get subtree
            tree_node.false_brunch = self.create_tree(data_list2, label_list2, best_attr_idx, -1, feature_idx_list)
        else:  # s2 is empty
            result = self.get_majority_label(label_list)
            tree_node.false_brunch = TreeNode(is_leaf=True, result=result, attr_idx=best_attr_idx, attr_val=-1)

        return tree_node

    def find_best_split(self, data_list, label_list, feature_idx_list):
        """ Returns the best split by GINI index

        :param data_list: subset of training data
        :param label_list: subset of training label
        :param feature_idx_list: list of feature indexes not split
        :type data_list: List[List[int]]
        :type label_list: List[int]
        :type feature_idx_list: List[int]
        :returns: split feature index and value of the best split
        :rtype: int, int
        """
        best_gini = 0.5
        best_feature_idx = -1
        best_feature_val = -1
        for feature_idx in feature_idx_list:
            feature_dict = self.feature_dict_list[feature_idx]
            min_gini = 0.5
            feature_val = -1
            if feature_idx in self.continuous_features:  # continuous features
                dict_len = len(feature_dict)
                cnt = 0
                for val in feature_dict.keys():
                    if cnt == dict_len - 1:  # last val no need to split
                        break
                    cnt += 1
                    data_list1, label_list1, data_list2, label_list2 = self.split_dataset(
                        data_list, label_list, feature_idx, val, is_continuous=True)
                    gini = self.cal_split_gini(data_list1, label_list1, data_list2, label_list2)
                    if gini < min_gini:
                        min_gini = gini
                        feature_val = val
            else:   # category features
                for val in feature_dict.keys():
                    data_list1, label_list1, data_list2, label_list2 = self.split_dataset(
                        data_list, label_list, feature_idx, val, is_continuous=False)
                    gini = self.cal_split_gini(data_list1, label_list1, data_list2, label_list2)
                    if gini < min_gini:
                        min_gini = gini
                        feature_val = val
            if min_gini < best_gini:
                best_gini = min_gini
                best_feature_idx = feature_idx
                best_feature_val = feature_val
        # print("best gini = " + str(best_gini) + " best_feature_idx = " + str(best_feature_idx) +
        #       " best_feature_val = " + str(best_feature_val))
        return best_feature_idx, best_feature_val

    @staticmethod
    def is_same_class(label_list):
        """ Check if labels belong to the same class

        :param label_list: subset of training label
        :type label_list: List[int]
        :return: -1 labels are NOT the same class,  >=0 the only label
        :rtype: int
        """
        first_label = label_list[0]
        for label in label_list:
            if first_label != label:
                return -1
        return first_label

    @staticmethod
    def is_same_attribute(data_list):
        """ Check if all data have the same attributes

        :param data_list: subset of training data
        :type data_list: List[List[int]]
        :rtype: bool
        """
        first_row = data_list[0]
        for row in data_list:
            if not row == first_row:
                return False
        return True

    @staticmethod
    def get_majority_label(label_list):
        """ Returns the majority label

        :param label_list: subset of training label
        :type label_list: List[int]
        :return: label that appear most frequently
        :rtype: int
        """
        return max(label_list, key=label_list.count)

    @staticmethod
    def split_dataset(data_list, label_list, attr_idx, attr_val, is_continuous=False):
        """ Split dataset into two subsets

        :param data_list: subset of training data
        :param label_list: subset of training label
        :param attr_idx: split attribute index
        :param attr_val: split attribute value
        :param is_continuous: if continuous attribute type
        :type data_list: List[List[int]]
        :type label_list: List[int]
        :type attr_idx: int
        :type attr_val: int
        :type is_continuous: bool
        :returns: subset of data_list & subset of label_list
        :rtype: List[List[int]], List[int], List[List[int]], List[int]
        """
        data_list1 = []
        data_list2 = []
        label_list1 = []
        label_list2 = []
        for data, label in zip(data_list, label_list):
            if is_continuous:
                if data[attr_idx] <= attr_val:
                    data_list1.append(data)
                    label_list1.append(label)
                else:
                    data_list2.append(data)
                    label_list2.append(label)
            else:  # category feature
                if data[attr_idx] == attr_val:
                    data_list1.append(data)
                    label_list1.append(label)
                else:
                    data_list2.append(data)
                    label_list2.append(label)
        return data_list1, label_list1, data_list2, label_list2

    def cal_split_gini(self, data_list1, label_list1, data_list2, label_list2):
        """ Calculate GINI index of a split

        :param data_list1: data subset1
        :param label_list1: label subset1
        :param data_list2: data subset2
        :param label_list2: label subset2
        :type data_list1: List[List[int]]
        :type label_list1: List[int]
        :type data_list2: List[List[int]]
        :type label_list2: List[int]
        :return: GINI index of the split
        :rtype: float
        """
        s1 = len(data_list1)
        s2 = len(data_list2)
        s = s1 + s2
        gini1 = self.cal_gini(data_list1, label_list1)
        gini2 = self.cal_gini(data_list2, label_list2)
        return round(s1 / s * gini1 + s2 / s * gini2, 6)

    @staticmethod
    def cal_gini(data_list, label_list):
        """ Calculate the GINI index of a subset

        :param data_list: subset of training data
        :param label_list: subset of training label
        :type data_list: List[List[int]]
        :type label_list: List[int]
        :return: GINI index of the subset
        :rtype: float
        """
        n = len(data_list)
        if n == 0:
            return 1
        ny = len(list(filter(lambda x: x == 1, label_list)))
        py = ny / n
        pn = 1 - py
        return round(1 - (py * py + pn * pn), 6)

    def classify(self, data):
        """ Classify a list of data

        :param data: list of data for classification
        :type data: List[List[int]]
        :return: predicted label 0 or 1
        :rtype: int
        """
        return self.traverse(self.root, data)

    def traverse(self, root, data):
        """ Traverse DecisionTree from certain node

        :param root: start treenode of the traverse
        :param data: list of data for classification
        :type root: TreeNode
        :type data: List[List[int]]
        :return: 0 or 1 label   -1 error or root is None
        :rtype: int
        """
        if root is None:
            return -1
        if root.is_leaf:
            return root.result
        if root.true_brunch:
            attr_idx = root.true_brunch.attr_idx
            attr_val = root.true_brunch.attr_val
            if data[attr_idx] == attr_val:
                return self.traverse(root.true_brunch, data)
        if root.false_brunch:
            return self.traverse(root.false_brunch, data)
        return -1

    def print_tree(self):
        output = open("decision_tree.txt", 'w+')
        print(self._tree(self.root), file=output)
        output.close()

    def _tree(self, point, prefix=[]):
        """ Create text description for decision tree

        :param point: treenode
        :param prefix: used in format
        :return: Text form of decision tree
        """
        space = '     '
        # branch = '│   '
        # tee = '├─ '
        # last = '└─ '
        branch = '|   '
        tee = '|- '
        last = '|_ '
        feature = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']
        feature_map = [
            ['< 30', '< 40', '< 50', '< 60', '> 60'],
            ["whether Without-pay", "whether State-gov", "whether Self-emp-not-inc", "whether Local-gov", "whether Federal-gov", "whether Private",
             "whether Self-emp-inc"],
            ["< 600000", "> 600000"],
            ["Prof-school", "Some-college", "5th-6th", "HS-grad", "11th", "9th", "Doctorate", "Assoc-acdm", "Preschool",
             "Masters", "7th-8th", "Assoc-voc", "10th", "12th", "1st-4th", "Bachelors"],
            ["< 11", "< 15", "> 15"],
            ["whether Married-spouse-absent", "whether Never-married", "whether Widowed", "whether Divorced", "whether Married-AF-spouse",
             "whether Married-civ-spouse", "Separated"],
            ["whether Sales", "whether Tech-support", "whether Craft-repair", "whether Armed-Forces", "whether Protective-serv",
             "whether Handlers-cleaners", "whether Other-service", "whether Machine-op-inspct", "whether Farming-fishing",
             "whether Priv-house-serv", "whether Adm-clerical", "whetherTransport-moving", "whether Prof-specialty", "whether Exec-managerial"],
            ["whether Not-in-family", "whether Unmarried", "whether Husband", "whether Own-child", "whether Wife", "whether Other-relative"],
            ["whether White", "whether Other", "whether Black", "whether Amer-Indian-Eskimo", "whether Asian-Pac-Islander"],
            ["whether Female", "whether Male"],
            ["< 5000", "< 10000", "< 15000", "< 20000", "> 20000"],
            ["< 1000", "< 1500", "< 2000", "> 2000"],
            ["< 20", "< 40", "> 40"]
        ]

        if len(prefix) == 0:
            self.tree_plot = ''

        self.tree_plot += ''.join(prefix)

        if not point.is_leaf:
            if point.true_brunch:
                self.tree_plot += "The splitting feature is " + str(
                    feature[point.true_brunch.attr_idx]) + " " + str(
                    feature_map[point.true_brunch.attr_idx][point.true_brunch.attr_val]) + "\n"
        else:
            if point.result == 0:
                self.tree_plot += "Label is <= 50K\n"
                # print("Label is <= 50K,", file=output, end='\t')
            else:
                self.tree_plot += "Label is > 50K\n"

        if len(prefix) > 0:
            prefix[-1] = branch if prefix[-1] == tee else space

        if not point.is_leaf:
            if point.true_brunch:
                self._tree(point.true_brunch, prefix + [tee])
            if point.false_brunch:
                self._tree(point.false_brunch, prefix + [last])
        return self.tree_plot
