import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("data/FinalBMD.csv")

#fracture column change to boolean

df['fracture'] = (df['fracture']=='fracture').astype(int)
df['sex'] = (df['sex']=='M').astype(int)

mean_of_bmd = df['bmd'].mean()
std_of_bmd = df['bmd'].std()

df['T-Score'] = (df['bmd']-mean_of_bmd)/(std_of_bmd/22.360679775)

df.to_csv('bmd_with_tscores',index=False)
final_df = pd.read_csv('bmd_with_tscores')

X = final_df[['age','sex','fracture','bmd']]
Y = final_df['T-Score']

Y_new = np.where(Y<=-2.5,1,0)

X_train,X_test,y_train,y_test = train_test_split(X,Y_new,test_size=0.3,random_state=42)
standard = StandardScaler()
train_features = standard.fit_transform(X_train)
test_features = standard.transform(X_test)
model = LogisticRegression()
model.fit(train_features, y_train)
train_score = model.score(train_features,y_train)
test_score = model.score(test_features,y_test)
y_pred = model.predict(test_features)

accuracy_of_model = accuracy_score(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)
class_report = classification_report(y_test,y_pred)