import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset_raw = pd.read_csv('train.csv')
list_cols = list(dataset_raw.columns.values)
print(list_cols)
dataset_raw_test = pd.read_csv('test.csv')

# Understanding dataset
dataset_train = dataset_raw.copy()
dataset_test = dataset_raw_test.copy()
print(dataset_train.info())
print(dataset_raw_test.info())
describe_train = dataset_train.describe()
describe_test = dataset_test.describe()

# Cleaning both datasets at once:
cleaner = [dataset_train, dataset_test]

print('Count of null values in training set:', dataset_train.isnull().sum())
print('Count of null values in test set:', dataset_test.isnull().sum())

# Dealing with missing values
# Dropping unncessary columns - PassengerID, Cabin, Ticket
drop_column = ['PassengerId', 'Cabin', 'Ticket']
dataset_train.drop(drop_column, axis = 1, inplace = True) 
dataset_test.drop(drop_column, axis = 1, inplace = True) 

# Dealing with missing values - Age, embarked, fare
for data in cleaner:
    data['Age'].fillna(data['Age'].mean(), inplace = True)
    data['Fare'].fillna(data['Fare'].mean(), inplace = True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    
print('Count of null values in training set:', dataset_train.isnull().sum())
print('Count of null values in test set:', dataset_test.isnull().sum())
# Missing values have been dealt with (can use fillna or imputer)

# Correlation of varibles to survival rate
correlation_matrix = dataset_train.corr()
correlation_matrix["Survived"].sort_values(ascending=False)

# Adding new features - Family size, whether passenger was alone and title
for data in cleaner:
    data['Familysize'] = data['SibSp'] + data['Parch'] + 1
    data['Alone'] = 1
    data['Alone'].loc[data['Familysize'] > 1] = 0 #If statement didn't work here for some reason
    data['Title'] = data['Name'].str.split(', ', expand = True)[1].str.split('.', expand = True)[0]
# Drop name from dataset
for data in cleaner:
     data.drop(['Name'], axis = 1, inplace = True)
# Another overview of data
print('Training set', dataset_train.isnull().sum())
print('Test set', dataset_test.isnull().sum())
dataset_train['Title'].value_counts()


# no. siblings and family size < 5% correlation. could possibly omit.
# Before that, encode categorical variables and then visualise coefficient again
"""#Label encoder 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for data in cleaner:
    data['sex_encoded'] = labelencoder.fit_transform(data['Sex']).reshape(-1,1)
    data['embarked_encoded'] = labelencoder.fit_transform(data['Embarked']).reshape(-1,1)
    data['title_encoded'] = labelencoder.fit_transform(data['Title']).reshape(-1,1)

# Drop sex, embarked, title
for data in cleaner:
    data.drop(['Sex', 'Embarked', 'Title'], axis = 1, inplace = True)

# Reshape data
for data in cleaner:
    data['sex_encoded'] = data['sex_encoded'].reshape(-1, 1)
# One hot encoding categorical variables   
onehotencoder = OneHotEncoder(sparse = False)
for data in cleaner:
    data['sex_onehot'] = onehotencoder.fit_transform(data['sex_encoded'])
    data['embarked_onehot'] = onehotencoder.fit_transform(data['embarked_encoded'])
    data['title_onehot'] = onehotencoder.fit_transform(data['title_encoded'])"""


# Using pandas get dummies to obtain dummy variables for categorical variables. This will be used when actually applying classification algorithms
dummy_train = pd.get_dummies(data = dataset_train, columns = ['Sex', 'Embarked', 'Title'])
dummy_test = pd.get_dummies(data = dataset_test, columns = ['Sex', 'Embarked', 'Title'])

# Label encoding for dummy variables - label encoding but not onehotencoding for graphical analysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for data in cleaner:
    data['sex_encoded'] = labelencoder.fit_transform(data['Sex']).reshape(-1,1)
    data['embarked_encoded'] = labelencoder.fit_transform(data['Embarked']).reshape(-1,1)
    data['title_encoded'] = labelencoder.fit_transform(data['Title']).reshape(-1,1)
# Split dataset into features and label
y_train = dummy_train[['Survived']]
X_train = dummy_train.copy()
X_train = X_train.drop(['Survived'], axis = 1)

# Now let's perform exploratory data analysis both descriptive and graphical
# Pearson's correlation with new variables
correlation_matrix_new = dataset_train.corr()
print(correlation_matrix_new['Survived'].sort_values(ascending = False))

# Simple stats
dataset_train['Survived'].value_counts() 
# 549 died and 342. 61.6%, 38.4% split 
dataset_train['Sex'].value_counts()
# 577 male, 314 female
survived_sex = dataset_train.groupby('Sex')['Survived'].mean()
print(survived_sex)
survived_class = dataset_train.groupby('Pclass')['Survived'].mean()
print(survived_class)
# table of Class and sex using concat and groupby
data_age_class = pd.concat(
        [dataset_train.groupby(['Pclass', 'Sex'])['Survived'].mean(),
         dataset_train.groupby(['Pclass', 'Sex'])['Survived'].count()],
         axis = 1)
# count of men and women who died given class
count_died = dataset_train[dataset_train['Survived'] == 0].groupby(['Pclass', 'Sex'])['Survived'].count()   
# count of men and women who survived
count_survived = dataset_train[dataset_train['Survived'] == 1].groupby(['Pclass', 'Sex'])['Survived'].count()   
# concat both datasets
data_age_class = pd.concat([data_age_class, count_survived, count_died], axis = 1)
  
data_cols = ['Survived', ' Total count', 'Total survived', 'Total died']
data_age_class.columns = data_cols   
# From table we can see that survival rate decreases as class goes up                  

correlationmatrix = dummy_train.corr(method = 'pearson')
print(correlationmatrix['Survived'].sort_values(ascending = False))
# Gender, titles, fare largest +- correlations - Scatter matrix to compare correlation between attributes
from pandas.tools.plotting import scatter_matrix
attributes = ['Sex_female', 'Fare', 'Embarked_C', 'Sex_male', 'Pclass', 'Alone']
scatter_matrix(X_train[attributes], figsize = (12, 8))

# Boxplots - age and fare
plt.boxplot(x = X_train['Age'], showmeans = True, meanline = True)
plt.title('Age boxplot')
plt.ylabel('Age (years)')

plt.boxplot(x = X_train['Fare'], showmeans = True, meanline = True)
plt.title('Fare boxplot')
plt.ylabel('Fare')

# Histograms - age and fare
plt.hist(x = [dataset_train[dataset_train['Survived'] == 1]['Age'],
         dataset_train[dataset_train['Survived'] == 0]['Age']],
         stacked = True, color = ['b', 'r'], label = ['Survived', 'Dead'])
plt.title('Histogram of age by survival')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.legend()

plt.hist(x = [dataset_train[dataset_train['Survived'] == 1]['Fare'],
         dataset_train[dataset_train['Survived'] == 0]['Fare']],
         stacked = True, color = ['b', 'r'], label = ['Survived', 'Dead'])
plt.title('Histogram of age by Fare')
plt.xlabel('Fare)')
plt.ylabel('Frequency')
plt.legend()

# Bar plots
# Bar plot of sex vs survival
sns.barplot(x = 'Sex', y = 'Survived', data = dataset_train, ci = None).set_title('Bar plot of sex vs survival rate')
# before plotting get an idea of how many passengers belong to each class
print(dataset_train['Pclass'].value_counts())
# Of the 891 passengers, 55.1% class 3, 24.2% class 2, 20.7% class 1.    
sns.barplot(x = 'Pclass', y = 'Survived', data = dataset_train, ci = None).set_title('Bar plot of class against survival')
sns.barplot(x = 'Embarked', y = 'Survived', data = dataset_train, ci = None).set_title('Bar plot of location of embarkment against survival rate')
# Bar plot sex vs survival given class.
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=dataset_train, ).set_title('Bar plot of sex, given class, against survival rate')
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data = dataset_train).set_title('Bar plot of sex, given embarkment, against survival rate')

# Plot of three graphs given embarkment location, sex, class and survival
embark = sns.FacetGrid(dataset_train, row = 'Embarked')
embark.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci = None)
embark.add_legend()
# Survival rate was higher for men who embarked from C = Cherbourg

# Different number of features in training data compared to test data. therefore have to drop columns
dummy_validation = dummy_test.copy()
dummy_validation.columns
dummy_train.columns
cols_to_drop = ['Title_Jonkheer', 'Title_Lady', 'Title_Capt', 'Title_Major', 'Title_Mlle', 'Title_Mme', 'Title_Sir', 'Title_the Countess']
dummy_train = dummy_train.drop(cols_to_drop, axis = 1)

# Split training set into training set and test set
X = dummy_train.drop(['Survived'], axis = 1)
y = dataset_train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

# Have to scale data for SVM
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)
X_validation = sc.fit_transform(dummy_validation)

# Fitting classifiers to training data
# SVM
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_svm.fit(X_sc, y_train)

# Random forest
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
classifier_RF.fit(X_train, y_train)

# Predicting with classifiers
y_pred_svm = classifier_svm.predict(X_test_sc)
y_pred_RF = classifier_RF.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test, y_pred_svm)
cm_RF = confusion_matrix(y_test, y_pred_RF)

from sklearn.metrics import accuracy_score
accuracy_SVM = accuracy_score(y_test, y_pred_svm)
accuracy_RF = accuracy_score(y_test, y_pred_RF)


# plot confusion matrix (from sklearn example)
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

classes = ['Died', 'Survived']
plot_confusion_matrix(cm_SVM, classes = classes, normalize = False)
plot_confusion_matrix(cm_RF, classes = classes, normalize = False)


# SVM has slightly higher accuracy score 81.6% vs 81.2%
# Can improve model accuracy using GridSearchCV - then use best model on validation data

from sklearn.model_selection import GridSearchCV

# SVM
parameters_SVM = [{'C':[1, 10, 100, 1000], 'kernel': ['linear'],
               'C':[1, 10, 100, 1000], 'kernel': ['rbf'],
               'gamma': [0.5, 0.1, 0.01, 0.001]}]
GS_SVM = GridSearchCV(estimator = classifier_svm, param_grid = parameters_SVM, scoring = 'accuracy',
                      cv = 10)
grid_search_SVM = GS_SVM.fit(X_sc, y_train)
best_accuracy_svm = grid_search_SVM.best_score_
best_parameters_svm = grid_search_SVM.best_params_

# RF
parameters_RF = {'bootstrap': [True, False],
                 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                 'max_features': ['auto', 'sqrt'],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 5, 10],
                 'n_estimators': [10, 50, 75, 100]}
GS_RF = GridSearchCV(estimator = classifier_RF, param_grid = parameters_RF, scoring = 'accuracy', cv = 10)
grid_search_RF = GS_RF.fit(X_train, y_train)
best_accuracy_RF = grid_search_RF.best_score_
best_parameters_RF = grid_search_RF.best_params_

# Now using new model parameters and fitting to training data
svm_classifier = SVC(kernel = 'rbf', gamma = 0.01, C = 10)
svm_classifier.fit(X_sc, y_train)

rf_classifier = RandomForestClassifier(n_estimators = 50, max_depth = 10, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 10)
rf_classifier.fit(X_train, y_train)

# Predicting test set with new models
y_pred_new_svm = svm_classifier.predict(X_test_sc)
y_pred_new_rf = rf_classifier.predict(X_test)

# confusion matrix
cm_new_SVM = confusion_matrix(y_test, y_pred_new_svm)
cm_new_RF = confusion_matrix(y_test, y_pred_new_rf)

# Accuracies
accuracy_SVM_new = accuracy_score(y_test, y_pred_new_svm)
accuracy_RF_new = accuracy_score(y_test, y_pred_new_rf)
print(accuracy_SVM_new)
print(accuracy_RF_new)

# Plot CM
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

classes = ['Died', 'Survived']
plot_confusion_matrix(cm_new_SVM, classes = classes, normalize = False)
plot_confusion_matrix(cm_new_RF, classes = classes, normalize = False)

# Using RF model on validation data (original test set) - predicting values
# Labelled dummy validation

y_pred_validation = rf_classifier.predict(dummy_validation)

# Creating submission file - Required columns are 'PassengerID' & 'Survived'
df_pred_validation = pd.DataFrame(y_pred_validation)
passengerID = dataset_raw_test['PassengerId'] 
submission = pd.concat([passengerID, df_pred_validation], axis = 1)
submission.columns = ['PassengerId', 'Survived']
# Saving predicted array values to CSV
np.savetxt('submission_1.csv', submission, delimiter = ',')

test = pd.read_csv('submission_1.csv')

submission.to_csv('submission_1.csv', index = False)
