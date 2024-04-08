sklearn model compare muliple models

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix
# import 
import numpy as np
import pandas as pd
# data = pd.concat([train,test])
df = pd.read_csv("sxx.csv", usecols=[1,2,3,4])
## explore null, missing
df.head(2)
df.isnull().head() 
df.isnull().sum()
# df = df.set_index('msisdn')
dem['date'] = pd.to_datetime(dem['date'])
dem['year'] = dem['date'].dt.year
dem['month'] = dem['date'].dt.month
dem['week'] = dem['date'].dt.weekofyear
dem['dayofyear'] = dem['date'].dt.dayofyear
dem['dayofmonth'] = dem['date'].dt.day
dem['dayofweek'] = dem['date'].dt.dayofweek

####################################
# Transform categorical features into numerical features
relevent_experience_map = {
    'Has relevent experience':  1,
    'No relevent experience':    0
}
experience_map = {
    '<1'      :    0,
    '1'       :    1, 
    '2'       :    2, 
    '3'       :    3
} 
def encode(df_pre):
    df_pre.loc[:,'relevent_experience'] = df_pre['relevent_experience'].map(relevent_experience_map)
    df_pre.loc[:,'experience'] = df_pre['experience'].map(experience_map)
    return df_pre
df = encode(df)

####################################


## graph
####################################
print('Train shape:', twosigma_train.shape)
print(twosigma_train.head())
print(twosigma_train.columns.tolist())
twosigma_train.var1.value_counts()
# Describe the train data
twosigma_train.describe()
####################################
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Find the median int by the label 
prices = twosigma_train.groupby('interest_level', as_index=False)['price'].median()
# Draw a barplot
fig = plt.figure(figsize=(7,5))
plt.bar(prices.interest_level, prices.price, width=0.5, alpha=0.8)
#Set titles
plt.xlabel('Interest level')
plt.ylabel('Median price')
plt.title('Median listing price across interest level')
plt.show()
####################################
# Draw a scatterplot
plt.scatter(x=train['fare_amount'], y=train['distance_km'], alpha=0.5)
plt.xlabel('Fare amount')
plt.ylabel('Distance, km')
plt.title('Fare amount based on the distance')
# Limit on the distance
plt.ylim(0, 50)
plt.show()
####################################
import matplotlib.pyplot as plt
plt.style.use('ggplot')
prices = df.groupby('label', as_index=False)['price'].median()
fig = plt.figure(figsize=(5,6))
plt.xlabel('')
plt.ylabel('')
print('test')
import pandas
import researchpy as rp
import seaborn as sns
var = 'inte_standard_label_health_health_cat'
table, results = rp.crosstab(df['new_segment1'], df[var], prop= 'col', test= 'chi-square')
results  
####################################
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

data1_pd = data1.toPandas()
corr = data1_pd.corr()
kot = corr[corr>=.5]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Greens")
####################################
cat_vars = df.select_dtypes(include=[object])
cat_vars.shape
cat_vars.columns

cat_vars = ["instant_bookable", "is_business_travel_ready", "cancellation_policy","host_is_superhost", "neighbourhood_cleansed","property_type","room_type", "bed_type"]
num_vars = ['price',  "number_of_reviews"]

from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('add_variables', NewVariablesAdder()),
    ('std_scaler', StandardScaler())
])
num_transformed = num_pipeline.fit_transform(df)

from sklearn.compose import ColumnTransformer
data_pipeline = ColumnTransformer([
    ('numerical', num_pipeline, num_vars),
    ('categorical', OneHotEncoder(), cat_vars),
])
airbnb_processed = data_pipeline.fit_transform(df)
df = pd.concat([num_transformed,airbnb_processed])
####################################
X = df[['โทรทัศน์','วิทยุ','หนังสือพิมพ์']]
y = df[['ยอดขาย']]

from sklearn.model_selection import train_test_split
X,X_test,y,y_test = train_test_split(X,y,train_size=0.7,random_state=7)


####################################
clfs = [KNeighborsClassifier(), 
        GaussianNB(),
        DecisionTreeClassifier(splitter = 'random'),
        RandomForestClassifier(),
        LogisticRegression(),
        LinearDiscriminantAnalysis(),
        GradientBoostingClassifier(), 
        MLPClassifier(),
        XGBClassifier(),
        LinearSVC(),
       ]
clfName = {0:'KNN', 1:'Naive Bayes', 2:'Decision Trees', 3:'Random Forests', 4:'LR', 5:'LDA', 
           6:'GradientBoost', 7:'MLP', 8:'XGBoost', 9:'SVC' }

## change threshold
threshold = 0.035
for i in range(len(clfs)):
    # scores = cross_val_score(clfs[i], X, y, cv=5, scoring = 'f1')
    # print (scores)
    clf = clfs[i].fit(X, y)
    y_pred = (clf.predict_proba(X_test)[:, 1] > threshold).astype('float')
    f1 = f1_score(y_test, y_pred)  
    print ('Test F1 for',clfName[i], ":", f1)
    print(f'Precision Score: {clfName[i]} {precision_score(y_test, y_pred)}') 
    print(f'Recall Score: {clfName[i]} {recall_score(y_test, y_pred)}')
    print(f'F1 Score: {clfName[i]} {f1_score(y_test, y_pred)}') 
    print(f'Accuracy Score: {clfName[i]} {accuracy_score(y_test, y_pred)}') 
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    disp.plot()  
####################################
#### sklearn model metrix (change cuttoff)

threshold = 0.035
y_pred = (clf.predict_proba(X_test)[:, 1] > threshold).astype('float')
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()

from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix, classification_report
print(f'Precision Score: {precision_score(y_test, y_pred)}') 
print(f'Recall Score: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}') 
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}') 

confusion_matrix(y_train, model.predict(x))
print(classification_report(y_train, model.predict(x)))
####################################
