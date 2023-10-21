import csv


def process_dataset(file_path='./adult/adult.data'):
    cleaned = clean_data(file_path)
    data, feature_dict = encode(cleaned)
    train_data = map(lambda row: row[:-1], data)
    train_label = map(lambda row: row[-1], data)
    return list(train_data), list(train_label), feature_dict

def clean_data(file_path):
    # read the training data as 2d list
    with open('./adult/adult.data', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    data = data[:len(data)-1]  # There is a white space before each string. Strip white space
    print("The size of the data: ", len(data), len(data[0]))
    for i in range(len(data)):  # There is a white space before each string. Strip white space
        data[i] = [s.strip() for s in data[i]]
    cleaned = [row for row in data if not any(el == '?' for el in row)]  # remove entries that contain "?"
    [row.pop(13) for row in cleaned]  # remove the 'native-country' column
    print("The size of the data after preprocessing: ", len(cleaned), len(cleaned[0]))
    return cleaned


# return a list with elements replaced to discrete integers according to dictionary
def replace(my_list, lookup_dict):
    # assume we can decide category or continuous based on the list length
    # case 1: category
    if len(set(my_list))==len(lookup_dict):
        return [list(lookup_dict.keys())[list(lookup_dict.values()).index(x)] for x in my_list]
    # case 2: continuous
    else:
        n = len(lookup_dict)
        res = []
        for i in range(len(my_list)):
            for k,v in lookup_dict.items():
                if k==n-1:
                    res.append(k)
                    break
                if my_list[i]<=float(v):
                    res.append(k)
                    break
    return res

# convert a row to readable form based on the input feature dictionary
def make_readable(row, f_dict):
    readable = []
    for i in range(len(row)):
        readable.append(f_dict[i][row[i]])
    return readable

def encode(cleaned):
    # colnames = ['age','workclass','fnlwgt','education','education-num','marital-status',
    #             'occupation','relationship','race','sex','capital-gain','capital-loss',
    #             'hours-per-week','income']

    continuous_features = [0,2,4,10,11,12]  # column index of features
    category_features = [1,3,5,6,7,8,9,13]
    columns = list(zip(*cleaned))  # transpose rows to columns for the ease of list operations
    for i in continuous_features:  # convert continuous values from string to float
        columns[i] = [float(x) for x in columns[i]]

    # create lookup dictionary for categorical and continuous features
    category_dict = []
    continuous_dict=[]
    for i in category_features:
        lookup_values = set(columns[i])
        category_dict.append(dict(zip(range(len(lookup_values)),lookup_values)))
    # age: <=20,21-30,31-40,41-50,51-60,61+
    continuous_dict.append(dict(zip(range(6),['20','30','40','50','60','60+'])))
    # fnlwgt(in 1e6): <=0.1, 0.1-0.2, ..., 0.6+
    continuous_dict.append(dict(zip(range(7),['100000','200000','300000','400000','500000','600000','600000+'])))
    # education-num: <=8, 8-9, 9-10, ..., 15-16
    continuous_dict.append(dict(zip(range(9),['8','9','10','11','12','13','14','15','16'])))
    # captital-gain: <=5000, 5000-10000, 10000-15000, 15000-20000, 20000+
    continuous_dict.append(dict(zip(range(5),['5000','10000','15000','20000','20000+'])))
    # capital-loss: <=1000, 1000-1500, 1500-2000, 2000+
    continuous_dict.append(dict(zip(range(4),['1000','1500','2000','2000+'])))
    # hours-per-week: <=10, 10-20, 20-30, 30-40, 40-50, 50-60, 60+
    continuous_dict.append(dict(zip(range(7),['10','20','30','40','50','60','60+'])))

    # create full list of feature dictionary, transform data accordingly
    all_features = continuous_features+category_features
    all_dict = continuous_dict+category_dict
    feature_dict = [d for idx,d in sorted(zip(all_features, all_dict))]
    for i in range(len(feature_dict)):
        columns[i] = replace(columns[i], feature_dict[i])
    backToRows = list(zip(*columns))  # transpose columns back to rows
    backToRows = [list(row) for row in backToRows]
    print("Feature dictionary is created. Data ready to use.")
    # example
    # print("Data size: \n", len(backToRows), len(backToRows[0]))
    # print("Data before transform: \n", cleaned[0])
    # print("Data after transform: \n", backToRows[0])
    # print("Convert to readable form using feature dictionary: \n",make_readable(backToRows[0], feature_dict))
    # print(feature_dict)
    return backToRows, feature_dict


if __name__ == '__main__':
    process_dataset()