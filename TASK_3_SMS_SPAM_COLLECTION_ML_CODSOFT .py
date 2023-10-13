#!/usr/bin/env python
# coding: utf-8

# # TASK 3  Machine Learning Intern @Codsoft 

# # 📚 Spam Classifier: SMS 📩
# Python · Spam SMS Collection Dataset 
# OBJECTIVES:- Build an AI model that can classify SMS messages as spam or legitimate. Use techniques like TF-IDF or word embeddings with classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines to identify spam messages
# 
# 

# # 1) Import Dependencies 📦📚 

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


get_ipython().system('pip install wordcloud')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import  Counter
import nltk
import pickle
import string 



get_ipython().run_line_magic('matplotlib', 'inline')
nltk.download('punkt')
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore") 


# ## 2) Load Dataset 

# In[4]:


# Importing Dataset 
data=pd.read_csv("spam_dataset_1.csv") 
data 


# In[5]:


# Specify the columns to drop
columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
# Use the drop method to remove the specified columns
data.drop(columns=columns_to_drop, inplace=True)
data 


# # 3) Data Preprocessing 🔄🔢 

# In[6]:


# Rename the Features
data.rename(columns={"v1": "Output","v2": "Input"}, inplace = True) 


# In[7]:


data.head(5) 


# 🏷️ Label Encoder 🏷️ 

# In[8]:


# Encoding Target Feature 
le = LabelEncoder()
data["Output"] = le.fit_transform(data["Output"]) 
data.head(5) 


# # 4) (EDA)📊 Exploratory Data Analysis 📊

# In[9]:


#Check shape of dataset 
data.shape 


# our dataset 5572 rows and 2 columns 

# In[10]:


#Check info our dataset 
data.info() 


# In[11]:


#Check describe our dataset
data.describe() 


# In[12]:


# Check Null Values
data.isnull().sum()  


# No null value in our dataset 

# In[13]:


# Check Duplicate Values
data.duplicated().sum() 


# 403 duplicated value in our dataset 

# In[14]:


# Drop Duplicates
data= data.drop_duplicates(keep="first") 
data 


# In[15]:


# Ham & Spam Counts
data["Output"].value_counts() 


# # 🥧 Pie Chart 📈

# In[16]:


plt.pie(data["Output"].value_counts(),autopct = "%.2f", labels=['ham','spam'])
plt.show() 


# # Obsrevations:
# Data is Inblanced 
# Having Less Spam Messages in Dataset

# # 5)  Feature Engineering 🔧

# In[17]:


# Total No. of Characters in Data
data["characters"] = data["Input"].apply(len)
data.head(5) 


# In[18]:


# Total No. of Words in Data
data["word"] = data["Input"].apply(lambda x:len( nltk.word_tokenize(x))) 
data.head(5) 


# In[19]:


# Total No. of Sentence
data["sentence"] = data["Input"].apply(lambda x:len(nltk.sent_tokenize(x)))
data.head(5) 


# In[20]:


# Statistical Analysis of new features
data[["characters","word", "sentence"]].describe() 
data


# In[21]:


# Statistical Analysis for HAM Data
data[data["Output"]==0][["characters","word", "sentence"]].describe()
data 


# In[22]:


# Statistical Analysis for SPAM Data
data[data["Output"] ==1][["characters","word", "sentence"]].describe() 
data 


# # 📊 Histogram Plot 📊

# In[23]:


plt.figure(figsize=(10,7))
sns.histplot(data[data["Output"]==0]["characters"],label= "ham",color="green")
sns.histplot(data[data["Output"]==1]["characters"],label= "spam",color = "red")
plt.title("SPAM Vs HAM : Characters")
plt.legend()
plt.show() 


# # Observations:
# Ham Characters and Words are more than Spam

# In[24]:


plt.figure(figsize=(10,7))
sns.histplot(data[data["Output"]==0]["word"],label= "ham",color="green")
sns.histplot(data[data["Output"]==1]["word"],label= "spam",color = "red")
plt.title("SPAM Vs HAM : Word")
plt.legend()
plt.show() 


# # Observations:
# Ham Characters and Words are more than Spam

# # Pair Plot 🌌

# In[25]:


sns.pairplot(data,hue="Output") 


# # Observations:
# Outliers are present

# In[26]:


# Intilizing Porter Stemmer Class
ps = PorterStemmer()


# In[27]:


# This Function helps to get Ready!!!

def data1(text):
    text = text.lower()               #  Converts Text in Lower Case
    text = nltk.word_tokenize(text)   #  Breaks Text in Words 
    
    y = []
    for i in text:
        if i.isalnum():               #  Removing Special Characters
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:                    #  Removing Stopwords and Punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:                    #  Porter Stemmer removing unwanted words
        y.append(ps.stem(i))
        
    return " ".join(y) 


# In[28]:


# Checking data function 
data1("aai and papa are everything for me") 


# In[29]:


data.sample(2)


# In[30]:


# Transforming dataset using the "data" function in new column "new_data"
data["new_data"] = data["Input"].apply(data1)


# In[31]:


data.sample(5) 


# # 6) Modile Building 

# # 🌫️ Word Cloud ☁️

# In[32]:


wc = WordCloud(
    background_color=None,
    width=800,
    height=400
)


# In[33]:


# Wordcloud for SPAM
spam_wc = wc.generate(data[data["Output"] ==1]["new_data"].str.cat(sep=" "))

# Wordcloud for HAM
ham_wc = wc.generate(data[data["Output"] ==0]["new_data"].str.cat(sep=" ")) 


# In[34]:


# SPAM 
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation="bilinear")
plt.axis("off")
plt.show() 


# In[35]:


# HAM
plt.figure(figsize=(10, 5))
plt.imshow(ham_wc, interpolation="bilinear")
plt.axis("off")
plt.show() 


# In[36]:


# Spliting SPAM Sentences in Words
spam_corpus = []
for msg in data[data["Output"]==1]["new_data"].tolist():
    for word in msg.split():
        spam_corpus.append(word)  


# # 📊 Bar Plot 📊 

# In[37]:


# Top 50 SPAM Words
a=pd.DataFrame(Counter(spam_corpus).most_common(50))[0]
b=pd.DataFrame(Counter(spam_corpus).most_common(50))[1]
plt.figure(figsize=(12,5))
sns.barplot(x= a,y=b)
plt.xticks(rotation=90)

plt.show() 


# In[38]:


# Spliting HAM Sentences in Words
ham_corpus = []
for msg in data[data['Output'] == 0]['new_data'].tolist():
    for word in msg.split():
        ham_corpus.append(word)  


# In[39]:


# Top 50 HAM Words
a=pd.DataFrame(Counter(ham_corpus).most_common(50))[0]
b=pd.DataFrame(Counter(ham_corpus).most_common(50))[1]
plt.figure(figsize=(12,5))
sns.barplot(x= a,y=b)
plt.xticks(rotation=90)

plt.show()


# # TFID 🔤 Vectorization 🔤

# In[40]:


# Initilizing TFIDF Vectorizer
tfidv = TfidfVectorizer(max_features=3000)
tfidv


# In[41]:


# Independent Feature
X = tfidv.fit_transform(data["new_data"]).toarray()
X 


# In[42]:


# Dependent Feature
y = data["Output"].values 
y 


# In[43]:


# Performing Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2) 


# # 🎓 Models Training 🤖

# In[44]:


# Models that are going to be trained
models={
    "Gaussian NB" : GaussianNB(),
    "Multinomial NB" : MultinomialNB(),
    "Bernoulli NB" : BernoulliNB(),
    "Logistic Regression" : LogisticRegression(),
    "SVC" : SVC(),
    "Decision Tree" : DecisionTreeClassifier(),
    "KNN" : KNeighborsClassifier(),
    "Bagging CLF" : BaggingClassifier(),
    "Random Forest" : RandomForestClassifier(),
    "ETC" : ExtraTreesClassifier(),
    "Ada Boost" : AdaBoostClassifier(),
    "Gradient Boost" : GradientBoostingClassifier(),
    "XGB" : XGBClassifier(),
    "XGBRF" : XGBRFClassifier()
} 


# In[45]:


# Creating a function train each model and calculate/return accuracy and precision
def train_clf (model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    
    return acc, pre 


# In[46]:


# A FOR loop Calls "traim_clf" for each model and stores accuracy and precision
acc_s=[]
pre_s=[]

for name, model in models.items():
    accuracy, precision = train_clf(model, X_train, y_train, X_test, y_test)
    
    acc_s.append(accuracy)
    pre_s.append(precision) 


# In[47]:


# As Precision matter over Accuracy in this Data, Sorting in DESC order of Precision. All Scores of Models

scores_data= pd.DataFrame({"Algorithm": models.keys(), 
                          "Accuracy": acc_s, 
                         "Precision": pre_s}).sort_values(by="Precision", ascending=False) 


# # 📈 Algorithms: Accuracy and Precision 🎯

# In[48]:


scores_data

##Observations:

Multinomial NB has an accuracy of 97.29% and a precision of 100.00%.
KNN (K-Nearest Neighbors) has an accuracy of 90.52% and a precision of 100.00%.
Bernoulli NB has an accuracy of 98.16% and a precision of 99.17%.
ETC (Extra Trees Classifier) has an accuracy of 97.97% and a precision of 98.35%.
Random Forest has an accuracy of 97.49% and a precision of 98.28%.
SVC (Support Vector Classifier) has an accuracy of 97.10% and a precision of 97.37%.
Logistic Regression has an accuracy of 95.16% and a precision of 96.81%.
Gradient Boost has an accuracy of 95.74% and a precision of 94.34%.
XGB (Extreme Gradient Boosting) has an accuracy of 97.10% and a precision of 94.26%.
Ada Boost has an accuracy of 96.42% and a precision of 93.16%.
Bagging CLF (Classifier) has an accuracy of 95.36% and a precision of 87.50%.
XGBRF (XGBoost Random Forest) has an accuracy of 94.00% and a precision of 87.25%.
Decision Tree has an accuracy of 94.39% and a precision of 80.77%.
Gaussian NB (Naive Bayes) has an accuracy of 86.75% and a precision of 50.22%. 
# # 📊 Scores Bar Plot 📊¶

# In[49]:


# Graph Accuracy and Precision

plt.figure(figsize=(10, 6))
bar_width = 0.35

plt.bar(scores_data["Algorithm"], scores_data["Precision"], width=bar_width, label="Precision", color='lightcoral', alpha=0.8)
plt.bar(scores_data["Algorithm"], scores_data["Accuracy"], width=bar_width, label="Accuracy", color='skyblue')
plt.xlabel("Algorithm")
plt.ylabel("Score")
plt.title("Accuracy and Precision Scores of Different Algorithms")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show() 


# # Observations:
# Models with high precision scores, such as Multinomial NB (100.00%), KNN (100.00%), and Bernoulli NB (99.17%), have a high ability to correctly identify positive instances.
# Models with high accuracy scores, such as Bernoulli NB (98.16%) and Random Forest (97.49%), make accurate overall predictions on the dataset.

# In[50]:


# Hence Multinomial Naïve Bayes give excellent precision and accuracy scores.
# According to me MNB is sutaible for Model

mnb=MultinomialNB()
mnb.fit(X_train, y_train) 


# # Pickle Files 📌

# In[51]:


# Pickle files help in Model Deployment

pickle.dump(mnb,open("model.pkl","wb"))
pickle.dump(tfidv, open("tfidf.pkl","wb"))
pickle.dump(data,open("fun.pkl","wb")) 


# In[ ]:




