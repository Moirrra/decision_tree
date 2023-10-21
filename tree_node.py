class TreeNode:
    def __init__(self, is_leave=False, result=-1, attr_idx=-1, attr_val=-1):
        self.true_brunch = None
        self.false_brunch = None
        self.is_leave = is_leave
        # label
        self.result = result
        # attribute index in dict
        self.attr_idx = attr_idx
        # attribute value
        self.attr_val = attr_val
