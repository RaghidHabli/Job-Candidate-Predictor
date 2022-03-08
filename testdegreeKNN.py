#Working Test degree knn
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

directory =r"C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\testDataDegreeKNN.csv"

data = pd.read_csv(directory)

##create the pickle file
with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name', 'wb') as f:
     pickle.dump(data, f)

##to load the pickle file
with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name','rb') as f:
    dest = pickle.load(f)

#Opens and reads the model
pick = open('knn_degreemodel.sav', 'rb')
model = pickle.load(pick)
pick.close()

features=[]
labels=[]
X=dest.iloc[:,:1].values
#Storing the column 1 in X and column 2 in  y
Y=dest.iloc[:,2:].values

labelencoder_degree= LabelEncoder()
inputs = dest
#FIX THE ENCODER for degrees repeat steps as before pop then add
inputs['degree_n'] = labelencoder_degree.fit_transform(inputs['Degree'])
fit = inputs.pop('Fit for the Job')
del inputs["Degree"]
inputs['Fit for the Job']=fit


for y in range(0,dest.shape[0]):

      features.append(inputs['degree_n'][y])

labels = Y
#print("X" , X)
# print("Y", Y)
print("labels",labels )
xtrain, xtest, ytrain, ytest =train_test_split(features,labels,test_size=0.2,shuffle=True)
#
print("Xtrain" , xtrain)
print("ytrain", ytrain)
#
#  #Testing phase: predict and store the predictions of the testing data in model_predictions
model_predictions = model.predict(np.reshape((xtest[0]),(1,-1)))


print ('ytest',ytest)
# #Another method used to calculate the accuarcy
accuracy = np.mean(model_predictions==ytest)
#
categories = ['Unfit', 'Fit'];
print('xtest at 0: ', np.reshape((xtest[0]),(1,-1)))
print('Accuracy: ', accuracy);
print('Prediction is: ', categories[model_predictions[0]]);