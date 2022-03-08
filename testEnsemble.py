import csv
import pickle

import pandas
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def createCSV(y, d):
    with open('mycsv.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)

        thewriter.writerow(['col1', 'col2'])

        for i in range(1, 100):
            thewriter.writerow(['one,two'])


def createDataset():
    directory = r"C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\newcsv.csv"
    data = pandas.read_csv(directory)
    with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name', "wb") as f:
        pickle.dump(data, f)

    ##to load the pickle file
    with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name', 'rb') as f:
        dest = pickle.load(f)

    # random.shuffle(dest)
    features = []
    labels = []
    labelencoder_degree = LabelEncoder()
    # print("Dest initial", dest)
    temp_dest = dest
    X = dest
    Y = temp_dest

    from sklearn.preprocessing import Binarizer
    Years = dest.iloc[:, :1].values
    bin = Binarizer(threshold=5)
    Years = bin.transform(Years)
    X['years_n'] = Years
    X['degree_n'] = labelencoder_degree.fit_transform(X['Degree'])

    del X['Fit']
    del X["Years"]
    del X["Position"]
    del X['Degree']
    del X['Skills']

    Y = data
    labelencoder_fitn = LabelEncoder()
    Y['fit_n'] = labelencoder_fitn.fit_transform(data['Fit'])
    del Y['Fit']
    del Y["Years"]
    del Y["Position"]
    del Y['Degree']
    del Y['Skills']
    for k in range(0, data.shape[0]):
        features.append([X['degree_n'][k], X['years_n'][k]])
        labels.append(Y['fit_n'][k])
    list1 = list()
    list1.append(features)
    list1.append(labels)
    return list1


def getPrediction(years, degree):
    list1 = createCSV()
    features = list1[0]
    labels = list1[1]
    pick = open('ensemblemodel.sav', 'rb')
    model = pickle.load(pick)
    pick.close()

    features, labels = createCSV()

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, random_state=1, stratify=labels,
                                                    shuffle=True)

    # model.fit(xtrain, ytrain)

    print("xtest: ", xtest)
    modelPredictions = model.predict(xtest)
    print("modelPred", modelPredictions)
    print(classification_report(ytest, modelPredictions))

