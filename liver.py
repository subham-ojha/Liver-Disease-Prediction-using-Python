# Importing the libraries

import pandas as pd 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Importing the dataset
dataset=pd.read_csv('Indian_Liver_Patient_Dataset.csv',header=None)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,10].values

# Handling missing data (9)
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,9:10])
X[:,9:10]=imputer.transform(X[:,9:10])

# Handling categorical variables
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])


# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 1, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)



def squared_error(actual, pred):
    return (pred - actual) ** 2
def check(actual, pred):

    if actual==pred:
        return 1
    else:
        return 0

def model(model):
    return model()

l=[MLPClassifier, KNeighborsClassifier , RandomForestClassifier]
for t in l:
    t1=time.time()
    clf=model(t)
    clf = clf.fit(X_train, y_train)
    time_taken = time.time() - t1
    predicted=clf.predict(X_test)
    error=0
    correct=0
    for i in range(len(X_test)):
        error+=squared_error(y_test[i],predicted[i])
        correct+=check(y_test[i],predicted[i])
    Mse=error/len(X_test)
    conf_mat=confusion_matrix(y_test, predicted)
    print("Time taken for {} is {}".format(t, time_taken))
    print(clf)
    print("For Model {} mean squared Error is {}".format(t,Mse))
    print("For Model {} Accuracy is {} percent".format(t,accuracy_score(y_test, predicted)))
    print("For Model {} F_Score is {} percent".format(t,f1_score(y_test, predicted, average="macro")))
    print("For Model {} Precision is {} percent".format(t,precision_score(y_test, predicted, average="macro")))
    print("For Model {} Recall is {} percent".format(t,recall_score(y_test, predicted, average="macro")))

