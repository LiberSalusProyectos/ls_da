# de https://www.kaggle.com/csyhuang/predicting-chronic-kidney-disease
# descripcion de variables https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease#
#reticulate::repl_python()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('G:/Mi Unidad/Desarrollo/datasets/kidney_disease_paraERC.csv')

# Map text to 1/0 and do some cleaning
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
df.rename(columns={'classification':'class'},inplace=True)

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
df['appet'] = df['appet'].replace(to_replace='no',value=0)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
df.drop('id',axis=1,inplace=True)

df2 = df.dropna(axis=0)
df2['class'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:,:-1], df2['class'], 
                                                    test_size = 0.33, random_state=44,
                                                   stratify= df2['class'] )
y_train = y_train.astype('int')
#y_train.value_counts()

tuned_parameters = [{'n_estimators':[7,8,9,10,11,12,13,14,15,16],'max_depth':[2,3,4,5,6,None],
                     'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[42]}]
clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10,scoring='f1')
clf.fit(X_train, y_train)

features = X_test.columns.values.tolist()
clf_best = clf.best_estimator_
importance = clf_best.feature_importances_.tolist()
feature_series = pd.Series(data=importance,index=features)
list_to_fill = X_test.columns[feature_series>0]

no_na = df2.index.tolist()
some_na = df.drop(no_na).apply(lambda x: pd.to_numeric(x,errors='coerce'))
some_na = some_na.fillna(0) # Fill up all Nan by zero.
X_test = some_na.iloc[:,:-1]
y_test = some_na['class']
lr_pred = clf_best.predict(X_test)

print('Accuracy: %3f' % accuracy_score(y_test, lr_pred))

from sklearn.externals import joblib
joblib.dump(clf, 'G:/Mi unidad/Desarrollo/desarrolloPython/ML4ERC_RF.pkl')

twoObs = pd.DataFrame(data=X_test.iloc[14:16,:], columns=df2.columns.values)
twoObs.to_csv('G:/Mi unidad/Desarrollo/datasets/twoObsERC.csv')

df_full = pd.read_csv('G:/Mi unidad/Desarrollo/datasets/twoObsERC.csv')
instancia_RF = joblib.load('G:/Mi unidad/Desarrollo/desarrolloPython/ML4ERC_RF.pkl')
format(instancia_RF.predict(df_full.iloc[:,0:24],))
