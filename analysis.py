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

# Import streamlit webapp framework
import streamlit as st

st.title('ML Classification Project: Extraa Learn')
st.markdown('*Part of Data Science & Machine Learning course from MIT IDSS*')

data = pd.read_csv('ExtraaLearn.csv')

st.header('Data',divider=True)
st.dataframe(data=data)
#print(data.head())
#print(data.shape)
#print(data.info())

st.header('Exploratory Data Analysis',divider=True)
#Prepare data for exploratory data analysis
data.drop(columns = 'ID', inplace=True) #Drop ID as it is only a unique identifier
data['time_spent_on_website'] = data['time_spent_on_website']/60 #Convert time spent on the site from seconds to minutes
cat_cols = data.select_dtypes(include = 'object').columns.tolist() #List out the categorical columns
num_cols = data.select_dtypes(include = ['int64','float64']).columns.tolist() #List out numerical columns
num_cols.remove('status') #Remove target variable from numerical columns list

def hist_box(data, col):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex='col', gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12, 6))
    # Adding a graph in each part
    # plt.suptitle(col)
    sns.boxplot(data=data, x=col, ax=ax_box, showmeans=True) #Boxplot
    sns.histplot(data=data, x=col, kde=True, ax=ax_hist) #Histogram
    ax_box.set_xticks([]) #Remove x-ticks
    ax_box.set_yticks([]) #Remove y-ticks
    ax_hist.axvline(data[col].mean(), color = 'green', linestyle = '--') #Mean indication line
    ax_hist.axvline(data[col].median(), color = 'black', linestyle = '-') #Median indication line
    return f

def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab = pd.crosstab(data[predictor], data[target], normalize = "index").sort_values(
        by = sorter, ascending = False
    )

    tab.plot(kind = "bar", stacked = True, figsize = (count + 1, 5))
    plt.legend(
        loc = "lower left",
        frameon = False,
    )
    plt.legend(labels = ['No','Yes'], loc = "upper left", bbox_to_anchor = (1, 1), title = 'Converted')
    plt.suptitle(target)
    f = plt.gcf()
    return f

#Numerical data analysis
st.subheader('Numerical')
selection_numCols = st.selectbox('Select a categorical variable',options=num_cols)
st.write(data[selection_numCols].describe())
fig_numCol = hist_box(data,selection_numCols)
st.pyplot(fig_numCol)
# with st.expander('Analysis'):
#     match selection_numCols:
#         case 'age':
#             st.markdown('Majority of the customers range from 35 to 60 with a mean age of 45 and a median of 50. Most of the customers however are aged older at 60 years or higher. This skew towards older ages is a point to be adressed. Adjusting marketing to bring in more younger customers closer towards the 18-25 years range will be beneficial.')
#         case 'website_visits':
#             st.markdown('Most customers only visit the website twice or thrice before making the decision to pay for the services. The website is mostly only being used as a way to explore and be introduced to the services. It is doubtful wether it could be used as a means to convert leads to paying customers purely through website without significant investment.')
#         case 'time_spent_on_website':
#             st.markdown('People only spend an average of 5 minutes on the website before making the decision. Though there are a significant number of people spending close to half an hour browsing the website, this pales in comparison to the number of people spending less than 10 minutes on the website. This enforces that the website is better used to market the services provided rather than for conversion.')
#         case 'page_views_per_visit':
#             st.markdown('Customers on average view only 2-4 pages per visit. Though there are quite a few outliers of people visiting more than 10 pages, this number is still much lower than those visting only a few pages.')

st.subheader('Categorical')
selection_catCols = st.selectbox('Select a categorical variable',options=cat_cols)
st.write(data[selection_catCols].value_counts())
fig_catCol = stacked_barplot(data=data,predictor=selection_catCols,target='status')
st.pyplot(fig_catCol)