class TreeNode:
    """ The definition of TreeNode

        :param is_leave: if leave node
        :param result: label of node
        :param attr_idx: attribute index of the split
        :param attr_val: attribute value of the split
        :type is_leave: bool
        :type result: int
        :type attr_idx: int
        :type attr_val: int
        """
    def __init__(self, is_leave=False, result=-1, attr_idx=-1, attr_val=-1):
        self.true_brunch = None
        self.false_brunch = None
        self.is_leave = is_leave
        self.result = result
        self.attr_idx = attr_idx
        self.attr_val = attr_val
