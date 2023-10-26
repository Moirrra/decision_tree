import csv
import json
import os.path


def process_dataset(file_path='adult/adult.data'):
    cleaned = clean_data(file_path)
    data, feature_dict, continuous_features, category_features = encode(cleaned)
    train_data = map(lambda row: row[:-1], data)
    train_label = map(lambda row: row[-1], data)
    return list(train_data), list(train_label), feature_dict, continuous_features, category_features


def clean_data(file_path):
    # read the training data as 2d list
    with open(file_path, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    if file_path.find('.test') != -1:
        data = data[1:len(data)-1]  # remove first text row
    else:
        data = data[:len(data)-1]  # There is a blank line in the end
    # print("The size of the data before preprocessing: ", len(data), len(data[0]))
    for i in range(len(data)):  # There is a white space before each string. Strip white space and remove dot
        data[i] = [s.strip().rstrip('.') for s in data[i]]
    cleaned = [row for row in data if not any(el == '?' for el in row)]  # remove entries that contain "?"
    [row.pop(13) for row in cleaned]  # remove the 'native-country' column
    # print("The size of the data after preprocessing: ", len(cleaned), len(cleaned[0]))
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


def encode(cleaned, file_path='my_dict.json'):
    # colnames = ['age','workclass','fnlwgt','education','education-num','marital-status',
    #             'occupation','relationship','race','sex','capital-gain','capital-loss',
    #             'hours-per-week','income']

    continuous_features = [0,2,4,10,11,12]  # column index of features
    category_features = [1,3,5,6,7,8,9,13]
    columns = list(zip(*cleaned))  # transpose rows to columns for the ease of list operations
    for i in continuous_features:  # convert continuous values from string to float
        columns[i] = [float(x) for x in columns[i]]

    if not os.path.exists(file_path):
        # print("Feature dictionary not found. Creating new...")
    # create lookup dictionary for categorical and continuous features
        category_dict = []
        continuous_dict=[]
        for i in category_features:
            lookup_values = set(columns[i])
            category_dict.append(dict(zip(range(len(lookup_values)),lookup_values)))
        # age: <=30,31-40,41-50,51-60,61+
        continuous_dict.append(dict(zip(range(5), ['30', '40', '50', '60', '60+'])))
        # fnlwgt(in 1e6): <=0.6, 0.6+
        continuous_dict.append(dict(zip(range(2), ['600000', '600000+'])))
        # education-num: <=11, 11-15, 15+
        continuous_dict.append(dict(zip(range(3), ['11', '15', '15+'])))
        # captital-gain: <=5000, 5000-10000, 10000-15000, 15000-20000, 20000+
        continuous_dict.append(dict(zip(range(5),['5000','10000','15000','20000','20000+'])))
        # capital-loss: <=1000, 1000-1500, 1500-2000, 2000+
        continuous_dict.append(dict(zip(range(4),['1000','1500','2000','2000+'])))
        # hours-per-week: <=20, 20-40, 40+
        continuous_dict.append(dict(zip(range(3), ['20', '40', '40+'])))

        # create full list of feature dictionary, transform data accordingly
        all_features = continuous_features+category_features
        all_dict = continuous_dict+category_dict
        feature_dict = [d for idx, d in sorted(zip(all_features, all_dict))]
        # # treat education as ordinal attribute
        # feature_dict[3] = {"0": "Preschool", "1": "1st-4th", "2": "5th-6th", "3": "7th-8th", "4": "9th","5": "10th", "6": "11th", "7": "12th", "8": "HS-grad", "9": "Assoc-voc", "10": "Assoc-acdm", "11": "Prof-school", "12": "Some-college", "13": "Bachelors", "14": "Masters", "15": "Doctorate"}
        # continuous_features = [0,2,3,4,10,11,12]  # column index of features
        # category_features = [1,5,6,7,8,9,13]
        with open('my_dict.json', 'w') as f:  # save the feature dictionary
            json.dump(feature_dict, f)
        # print("Feature dictionary is created.")
    else:
        # print("Found existing feature dictionary.")
        with open('my_dict.json') as f:
            feature_dict = json.load(f)
        for i in range(len(feature_dict)):  # convert keys from str to int
            feature_dict[i] = {int(k): v for k, v in feature_dict[i].items()}

    for i in range(len(feature_dict)):
        columns[i] = replace(columns[i], feature_dict[i])
    backToRows = list(zip(*columns))  # transpose columns back to rows
    backToRows = [list(row) for row in backToRows]



    print("Data ready to use.")
    # example
    # print("Data size: \n", len(backToRows), len(backToRows[0]))
    # print("Data before transform: \n", cleaned[0])
    # print("Data after transform: \n", backToRows[0])
    # print("Convert to readable form using feature dictionary: \n",make_readable(backToRows[0], feature_dict))
    # print(feature_dict)
    return backToRows, feature_dict, continuous_features, category_features
