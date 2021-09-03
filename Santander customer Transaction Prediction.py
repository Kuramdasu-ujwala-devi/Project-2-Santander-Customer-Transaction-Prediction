#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
print("---loading libariaries---")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np 

#Complementary error function
from scipy.special import erfc

#Random Sampling 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


#Feature Scaling, Extraction
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Plot
import matplotlib.pyplot as plt
import seaborn as sns

#Model Selection
from sklearn.model_selection import train_test_split

#Machine Learning Algorthims
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

#Metrics
from sklearn.metrics import roc_auc_score, plot_roc_curve, classification_report, confusion_matrix


#load data
df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")

#shape of train and test data
df_train.shape
df_test.shape

df_train.info()

#checking types
df_train.dtypes

#observing data
df_train.head(5)
df_test.head(5)

#counting observations per target class
df_train.target.value_counts()

#plotting pie chart for target class
df_train['target'].value_counts().plot(kind='pie', figsize=(5,5))

#checking for missing values in train data
df_train.isna().sum().sum()

#checking for missing values in train data
df_test.isna().sum().sum()

#Summary of train data
df_train.describe()


#checking outliers using Chauvenet's criterion
def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.    
    return prob < criterion       # Use boolean array outside this function

numerical_features=df_train.columns[2:]

#outliers in each variable in train data 
train_outliers = dict()
for col in [col for col in numerical_features]:
    train_outliers[col] = df_train[chauvenet(df_train[col].values)].shape[0]
train_outliers = pd.Series(train_outliers)

#Plotting outliers in a barplot
train_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');


#printing otliers percentage of train dataset
print('Total number of outliers in training set: {} ({:.2f}%)'.format(sum(train_outliers.values), (sum(train_outliers.values) / df_train.shape[0]) * 100))

#outliers in each variable in test data 
test_outliers = dict()
for col in [col for col in numerical_features]:
    test_outliers[col] = df_test[chauvenet(df_test[col].values)].shape[0]
test_outliers = pd.Series(test_outliers)

#plotting outliers in a barplot
test_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');

#printing outliers percentage in test dataset
print('Total number of outliers in testing set: {} ({:.2f}%)'.format(sum(test_outliers.values), (sum(test_outliers.values) / df_test.shape[0]) * 100))

#remove these outliers in train and test data
for col in numerical_features:
    train=df_train.loc[(~chauvenet(df_train[col].values))]
for col in numerical_features:
    test=df_test.loc[(~chauvenet(df_test[col].values))]    
    
#shape of train and test data after removal of outliers 
train.shape,test.shape

#describe train data after removal of outliers
train.describe()

#describe test data after removal of outliers
test.describe()

#hisograms are used to check distribution of data 
#draw histograms of numeric data in training set 
print("Distribution of Columns")
plt.figure(figsize=(40,200))
for i,col in enumerate(numerical_features):
    plt.subplot(50,4,i+1)
    plt.hist(train[col])
    plt.title(col)  

print("Almost all features follow fairly normal distribution ")


#defining confusion matrix
def cnf_mtx(ytest, ypred, filename):
    import seaborn as sns
    import matplotlib.pyplot as plt 
    from sklearn.metrics import confusion_matrix
    plt.clf()
    cm = confusion_matrix(ytest,ypred)
    print("----confusion matrix-----")
    print(cm)
    plt.title(filename.split(".")[0]) # title with fontsize 20

    plot = sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted labels') # x-axis label with fontsize 15
    plt.ylabel('True labels') # y-axis label with fontsize 15
    plot.get_figure().savefig(filename)
    

if __name__ == "__main__":
    print("Processing to machine learning algorithms by over sampling method ")

    print(str(train['target'].value_counts())+"\n\nCreating the X features and Y target variables..\n")

    X_train = train.drop(columns=['ID_code', 'target'])
    y_train = train['target']
    X_val = test.drop(columns='ID_code')

    #Fixing Class Imbalanced
    print("Found the Class Imbalanced \n\nFixing the Class imbalanced by over sampling with SMOTE.\n")

    randsmp =  SMOTE()
    X_train, y_train = randsmp.fit_resample(X_train, y_train)

    print("Creating a pipline for fit and transform features into standard scale and reducing features..\n")

    steps = [("Standarad Scalar", StandardScaler(with_mean=True)),
             ("PCA", PCA(n_components=10))
            ]

    pipline = Pipeline(steps=steps)
    pXtrain = pipline.fit_transform(X_train)
    

    #Split dataset into train and size of 25% in dataset as test data.
    print("Split dataset into Train and Test \n")

    Xtrain, Xtest, ytrain, ytest = train_test_split(pXtrain, y_train, test_size=0.25, random_state=42 )

    print("Now, Training the Machine Learning Model with Xtrain, ytrain. and Validating the Model by Xtest, ytest. \n")

    
    print("Logistic Regression:- ")
    #Logistic Regression
    lr = LogisticRegression()
    lr.fit(Xtrain, ytrain) ## GENETRATED WEIGHTS 

    lr_ypred = lr.predict(Xtest) # FINDING TARGETS

    #print classification report
    print(classification_report(ytest, lr_ypred)+"\n\n")
    #Print Accuracy
    print("AUC Score:- %.3f \n\n" % roc_auc_score(ytest, lr_ypred))
    
    #print confusion matrix
    cnf_mtx(ytest, lr_ypred, "LR-Confusion Matrix.png")
    #plot ROC Curve
    plot_roc_curve(lr, Xtest, ytest)
    #axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #show plot
    plt.show()
    
    
    print("RandomForest Classifier:- ")
    #RandomForest Classifer
    rnclf = RandomForestClassifier(n_estimators=5)
    rnclf.fit(Xtrain, ytrain)
    rnclf_ypred = rnclf.predict(Xtest)
    
    #Print Classification Reprt
    print(classification_report(ytest, rnclf_ypred)+"\n\n")
    #Print Accuracy
    print("AUC Score:- %.3f \n\n" % roc_auc_score(ytest, rnclf_ypred))
    
    #Print Confusion Matrix
    cnf_mtx(ytest, rnclf_ypred, "Rnclf-ConfusionMatrix.png")
    #plot Roc Curve
    plot_roc_curve(rnclf, Xtest, ytest)
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #show plot
    plt.show()
    
    
    print("LinearSVC Classifier:- ")
    #Linear SVC
    svc = LinearSVC()
    svc.fit(Xtrain, ytrain)
    svc_ypred = svc.predict(Xtest)
    
    #print classification report
    print(classification_report(ytest, svc_ypred)+"\n\n")
    #print Accuracy
    print("AUC Score:- %.3f \n\n" % roc_auc_score(ytest, svc_ypred))
    
    #print confusion matrix
    cnf_mtx(ytest, svc_ypred, "L SVC-ConfusionMatrix.png")
    #plot ROC curve
    plot_roc_curve(svc, Xtest, ytest)
    #axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #show plot
    plt.show()
    
    
    print("Gaussian NB:- ")
    #GaussianNB
    gnb = GaussianNB()
    gnb.fit(Xtrain, ytrain)
    gnb_ypred = gnb.predict(Xtest)
    
    #print classification report
    print(classification_report(ytest, gnb_ypred)+"\n\n")
    #print accuracy score
    print("AUC Score:- %.3f \n\n" % roc_auc_score(ytest, gnb_ypred))
   
    #print confusion matrix
    cnf_mtx(ytest, gnb_ypred, "GaussionNB-ConfusionMatrix.png")
    #plot ROC curve
    plot_roc_curve(gnb, Xtest, ytest)
    #axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #show plot
    plt.show()  


    #Beste Performance of ML Model is RandomForestClassifier and GaussianNB
    print("--- Best performance of ML model are RandomForestClassifier and GaussainNB---")
    print("--- As their accuracy score is almost equal ---")
    
    pX_val = pipline.transform(X_val)

    y_val = rnclf.predict(pX_val)
    y1_val = gnb.predict(pX_val)

    pd.DataFrame({'ID_code': test['ID_code'], 'target': y_val}).to_csv("submissionRFC-p.csv", index=False)
    pd.DataFrame({'ID_code': test['ID_code'], 'target':y1_val}).to_csv("submissionGNB-p.csv", index=False)
    
    print("--- End of the program ---")

