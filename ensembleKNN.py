#NEED FIXING

import pickle

import pandas
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def getDataset():
    directory = r"C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\DataTraining.csv"
    data = pandas.read_csv(directory)
    with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name',"wb") as f:
        pickle.dump(data, f)

    ##to load the pickle file
    with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name', 'rb') as f:
        dest = pickle.load(f)

    #random.shuffle(dest)
    features =[]
    labels = []
    labelencoder_degree = LabelEncoder()
    # print("Dest initial", dest)
    temp_dest = dest
    X = dest
    Y = temp_dest
    #X = dest.iloc[:, :2].values
    #dest.iloc[:,1 :2]

    from sklearn.preprocessing import Binarizer
    Years = dest.iloc[:, :1].values
    #
    # #print("Before: ", Y)
    bin = Binarizer(threshold=5)
    Years = bin.transform(Years)
    X['years_n'] = Years
    X['degree_n']= labelencoder_degree.fit_transform(X['Degree'])

    del X['Fit']
    del X["Years"]
    del X["Position"]
    del X['Degree']
    del X['Skills']
    #print("X", X)

    # print(X[0]) #year + degree
    # print(dest.iloc[:, :1]) #only years
    #print ("dest shape: " , dest.shape[0]) 54
    #Y = dest.iloc[:, 4:].values
    #print("Y", Y) #[[Yes],[No]]


    #HERE
    Y = data

    labelencoder_fitn = LabelEncoder()
    Y['fit_n'] = labelencoder_fitn.fit_transform(data['Fit'])
    del Y['Fit']
    del Y["Years"]
    del Y["Position"]
    del Y['Degree']
    del Y['Skills']
    #print("Y: ", Y)
    ##
    ## print("Y[0]", Y['fit_n'][0])
    ## Split the elements in data into features and labels
    ##print ("X[0]", X['degree_n'][0])
    for k in range(0,data.shape[0]):
           features.append([X['degree_n'][k],X['years_n'][k]])
           labels.append(Y['fit_n'][k])
    ##random.shuffle(features)
    # print("Features: ", features)
    # print("Labels: ", labels)
    list1 = list()
    list1.append(features)
    list1.append(labels)
    return list1

#getDataset()
list2 = getDataset()
features = list2[0]

labels = list2[1]
print("features: ", features)
print("\nlabels: ", labels)
#
# #Opens and reads the degree model
pick = open('knn_degreemodel.sav', 'rb')
model_degree = pickle.load(pick)
pick.close()

#Opens and reads the years model
pick2 = open('knn_model.sav','rb')
model_years = pickle.load(pick2)
pick2.close()

level0 = list()
level0.append(('knn1',model_years))
level0.append(('knn2',model_degree))

level1 = DecisionTreeClassifier()
model = StackingClassifier(estimators=level0, final_estimator=level1,cv=5)

features, labels = getDataset()

xtrain, xtest, ytrain, ytest = train_test_split(features,labels,test_size=0.2,random_state=1,stratify=labels,shuffle=True)

model.fit(xtrain, ytrain)

pick = open('ensemblemodel.sav', 'wb')
pickle.dump(model, pick)
pick.close()

print("xtest: ", xtest)
modelPredictions = model.predict(xtest)
print ("modelPred", modelPredictions)
print(classification_report(ytest,modelPredictions))