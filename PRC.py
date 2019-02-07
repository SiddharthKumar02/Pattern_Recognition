import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#Verified: No of records given in website matches with the no of records loaded into dataframe using pandas.read_csv
data = pd.read_csv('Spect Dataset/Spect Dataset/Dataset.csv', header = None)  #Scanning Dataset
data.head()
df = pd.DataFrame(data)

array = data.values
X = array[:, 1:23] #Independent columns
y = array[:, 0]     #Dependent column

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 500, test_size = 0.50)
X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, random_state = 500, test_size = 0.50)

Classifier = GaussianNB()
Classifier.fit(X_train, y_train)
Classifier.score(X_train, y_train) ## Score of training set = 0.5522388059701493
Classifier.fit(X_val, y_val)
Classifier.score(X_val, y_val)     ## Score of validation set = 0.9253731343283582
y_Predict = Classifier.predict(X_test)
Classifier.score(X_test, y_test)   ## Score of test set = 0.8656716417910447
data.hist()



for i in range(len(array[0])):
    sd = np.sqrt(df[i].var())
    mean = df[i].mean()
    count, bins, ignored = plt.hist(df[i], 14, density = True)
    plt.plot(bins, 1/(sd * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean)**2/(2 * sd**2)), linewidth = 3, color = 'green')
    plt.title('For column number' + str(i+ 1))
    plt.savefig('Images/col' + str(i) + '.png')
    plt.show()
    


for i in range(len(array[0])):
    print("Mean of "+ str(i+1) +", is "+str(df[i].mean()), file=open('/variance/'+str(i)+'.txt'))

print("\n")

for i in range(len(array[0])):
    print("Variance of "+ str(i+1) +", is "+str(df[i].var()), file=open('/variance/'+str(i)+'.txt'))
    
    
    
Matrix = confusion_matrix(y_test, y_Predict)
print(Matrix)

TP = Matrix[0][0]
print("TP: ",TP)
FP = Matrix[0][1]
print("FP: ",FP)
FN = Matrix[1][0]
print("FN: ",FN)
TN = Matrix[1][1]
print("TN: ",TN)


precision = (TP/TP+FP)
print("Precision: ", precision)
recall = (TP/TP+FN)
print("Recall: ", recall)
FMeasure = (2*precision * recall/precision + recall)
print("FMeasure", FMeasure)
MCC = (TP * TN - FP * FN)/(np.sqrt((TP + FP)*(TP + FN)*(TN + FN)*(TN + FP)))
print("MCC: ", MCC)
