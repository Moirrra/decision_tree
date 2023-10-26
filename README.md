## Group Members
| Name        | Student Number |
|-------------|----------------|
| TANG Shi    | 1155196621     |
| WANG Yiwen  | 1155198084     |
| ZHUANG Yuan | 1155198046     |


## Running the program
* Python 3.9+
* Run the command: `python main.py`

## Code structure
The main classes containing the logic of the codes are the following:
* **data_process.py**: Preprocessing the training and testing dataset including removing the meaningless records and features, dividing the continuous features into groups, and re-tag the categorical features.
* **decision_tree.py**: Main procedure of building decision tree.
* **tree_node.py**: Definition of TreeNode.
* **main.py**: Main code to start the program.

## Document Files in the folder
The document files in the folder are explained here in details of the functions.
* **output.xlsx**: the report on using the tree to classify the records of the evaluation set, including the each record's attributes in the evaluation set and whether it has been classified correctly in the last column.
* **project.pdf**: the report on explaining the decision tree and process of the executing program
* **decision_tree.txt**: the report of the constructed decision tree, which will be created and updated in every run.

