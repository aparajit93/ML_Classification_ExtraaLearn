# Import numerical and data analysis libraries
import numpy as np
import pandas as pd
# Import plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Import the Machine Learning models required from Scikit-Learn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Import the other functions required from Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV

# Import functions to get metric scores
from sklearn.metrics import confusion_matrix,classification_report

# Import pickle for saving models
import pickle

data = pd.read_csv('ExtraaLearn.csv')
#print(data.head())
#print(data.shape)
#print(data.info())

#Prepare data
data.drop(columns = 'ID', inplace=True) #Drop ID as it is only a unique identifier
data['time_spent_on_website'] = data['time_spent_on_website']/60 #Convert time spent on the site from seconds to minutes

df = data.copy() #Make a copy of the data set to work with

df.replace({'Yes':1, 'No':0},inplace=True) #Convert all yes and no to numeric/bool
df = pd.get_dummies(df, columns = ['current_occupation', 'first_interaction', 'profile_completed', 'last_activity']) #Convert ctegorical columns into numeric using dummy variables

#print(df.info()) #Check new dtypes

X = df.drop(columns = 'status') #Create independent variable
Y = df['status'] #Create dependent variable

# Splitting the dataset into train and test datasets in 70:30 split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle = True, random_state = 1, stratify = Y)
with open('ExtraaLearn_data_decisionTree.pkl', 'wb') as f:
    pickle.dump(x_train, f)
    pickle.dump(x_test, f)
    pickle.dump(y_train, f)
    pickle.dump(y_test, f)

#Defining precission, recall, f1 score and plotting the confusion matrix
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    
    sns.heatmap(cm, annot=True,  fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

#Defining a function to visualize decision trees
def DecisionTree_Visualizer(predictors,classifier,depth=None):
    feature_names = list(predictors.columns)
    plt.figure(figsize=(20, 10))
    out = tree.plot_tree(
        classifier,
        max_depth = depth,
        feature_names=feature_names,
        filled=True,
        fontsize=9,
        node_ids=False,
        class_names=True,
    )
    plt.show()

dt_class_model = DecisionTreeClassifier(random_state=1) #Create a decision tree model
dt_class_model.fit(x_train,y_train) #Training the tree on training data

DecisionTree_Visualizer(x_train,dt_class_model,depth = 3)

#Checking the performance of the decision tree to training data
dt_class_pred_train = dt_class_model.predict(x_train)
metrics_score(y_train, dt_class_pred_train)

#Checking the performance of the decision tree on the test data
dt_class_pred_test = dt_class_model.predict(x_test)
metrics_score(y_test, dt_class_pred_test)

dt_class_tuned = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(1,40,5), #Original tree has a depth of 40
    "max_leaf_nodes": [50, 75, 150, 250],
    "min_samples_split": [10, 20, 30, 50, 70],
}
# Run the grid search
grid_obj = GridSearchCV(dt_class_tuned, parameters, cv=5,scoring='recall',n_jobs=-1)
grid_obj = grid_obj.fit(x_train, y_train)

# Set the tree to the best combination of parameters
dt_class_tuned = grid_obj.best_estimator_

# Fit the best tree to the data.
dt_class_tuned.fit(x_train, y_train)

#Checking the performance of the pruned decision tree to training data
dt_class_pred_train_tuned = dt_class_tuned.predict(x_train)
metrics_score(y_train, dt_class_pred_train_tuned)

#Checking the performance of the pruned decision tree to test data
dt_class_pred_test_tuned = dt_class_tuned.predict(x_test)
metrics_score(y_test, dt_class_pred_test_tuned)

DecisionTree_Visualizer(x_train,dt_class_tuned,depth=4)

#List out the features on which the tree splits and their importances
importances = dt_class_tuned.feature_importances_

columns = x_train.columns

importance_dt = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)


plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
sns.barplot(x = importance_dt.Importance, y = importance_dt.index, color="violet")
plt.show()

with open("ExtraaLearn_decisionTree.pkl", "wb") as f:
    pickle.dump(dt_class_model, f)

with open("ExtraaLearn_decisionTree_tuned.pkl", "wb") as f:
    pickle.dump(dt_class_tuned, f)