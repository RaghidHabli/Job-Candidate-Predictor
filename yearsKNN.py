#Working KNN  years of experience

import numpy as np    #numpy is a library for making computations
import matplotlib.pyplot as plt    #it is a 2D plotting library
import pandas as pd    # pandas is mainly used for data analysis
import seaborn as sns    # data visualization library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import random
data=pd.read_csv("C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\Salary_Data.csv")

##to create the pickle file
with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name', 'wb') as f:
     pickle.dump(data, f)

##to load the pickle file
with open('C:\\Users\\lenovo\\PycharmProjects\\RecruitmentSystem\\source_object_name','rb') as f:
    dest = pickle.load(f)

print(dest)

features=[]   #nb of years
labels=[]     #fit or unfit for training


X=dest.iloc[:,:-1].values
#Storing the column 1 in X and column 2 in  y
Y=dest.iloc[:,:2].values

print("X",X)
print("Y",Y)
#snsdistplot(df['YearsExperience'],kde=False,bins=10)

#Years of Experience, level of degree, skills
for y in range(0,29):

     features.append(X[y][0])
     ##accuracies.append(np.mean(predictions == ytest))
     ##models.append(knn_model)
for y in range(0,29):
     if (X[y][0]>5):
        labels.append (1) #Senior
     else:
        labels.append(0) #Junior


print("YearsOfExperience: ", features)
print("Salary", labels)



xtrain, xtest, ytrain, ytest = train_test_split(features,labels,test_size=0.2,shuffle=True)

accuracies =[]
models= []

for k in range(1,8):
    knn_model=KNeighborsClassifier(n_neighbors=k)
    xtrain = np.array(xtrain).reshape(-1,1)
    xtest =np.array(xtest).reshape(-1,1)
    knn_model.fit(xtrain,ytrain)
    predictions = knn_model.predict(xtest)
    accuracies.append(np.mean(predictions==ytest))
    models.append(knn_model)

for k in range(1,8):
    print("Accuracy for k: ",k)
    print(accuracies[k-1])

# Plotting accuracies
plt.plot(range(1,8),accuracies)
plt.ylabel("Accuracies")
plt.xlabel("# of Neighbours")
plt.title("Accuracies")
plt.grid()
plt.show()
print("xtest: ", xtest)
print(predictions)

#print("Accuracy for k: " ,4)
#accuracies.append(np.mean(predictions==ytest))
#print(accuracies)


# #Plotting accuracies
plt.plot(range(1,8),accuracies)
plt.ylabel("Accuracies")
plt.xlabel("# of Neighbours")
plt.title("Accuracies")
plt.grid()
plt.show()
#
#Get accuracies as percentages
percentages=[]
for i in range(1,8):
        percentages.append(100*accuracies[i-1])

#Find maximum value and corresponding K index
maximum = max(percentages)
print("Max percentage: ", maximum)
print("K-value: ", percentages.index(maximum)+1)
print("K-value: ", accuracies.index(max(accuracies))+1)

index = percentages.index(maximum)+1
print(percentages[index-1])

plt.plot(range(1,8),percentages)
plt.ylabel("Accuracies Percentages")
plt.xlabel("# of Neighbours")
plt.title("Percentages")
plt.grid()
plt.show()

optimized_model = models[index-1]
#Save the model in 'model.sav' folder
pick = open('knn_model.sav', 'wb')
pickle.dump(knn_model, pick)
pick.close()





