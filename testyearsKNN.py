#Working Test Years KNN
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

directory =r"C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\TestData.csv"

data = pd.read_csv(directory)

##create the pickle file
with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name', 'wb') as f:
     pickle.dump(data, f)

##to load the pickle file
with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name','rb') as f:
    dest = pickle.load(f)

#Opens and reads the model
pick = open('knn_model.sav', 'rb')
model = pickle.load(pick)
pick.close()


features=[]
labels=[]
X=dest.iloc[:,:1].values
#Storing the column 1 in X and column 2 in  y
#Y=dest.iloc[:,:-1].values
Y=dest.iloc[:,2:].values


print("X" , X)
print("Y", Y)
#print("size of X", dest.shape[0])

#Years of Experience, level of degree, skills
for y in range(0,dest.shape[0]):

     features.append(X[y][0])



print("X" , X)
print("Y", Y)
labels = Y
xtrain, xtest, ytrain, ytest =train_test_split(features,labels,test_size=0.2,shuffle=True)

print("Xtrain" , xtrain)
print("ytrain", ytrain)
print ("xtest", xtest)
print ("ytest",ytest)
 #Testing phase: predict and store the predictions of the testing data in model_predictions
model_predictions = model.predict(np.reshape((xtest[0]),(1,-1)))

#Another method used to calculate the accuarcy
accuracy = np.mean(model_predictions==ytest)

categories = ['Unfit', 'Fit'];
print('xtest at 0: ', xtest[0])
print('Accuracy: ', accuracy);
print('Prediction is: ', categories[model_predictions[0]]);




