# de https://www.kaggle.com/mansoordaku/ckdisease/
# descripcion de variables https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease#
#reticulate::repl_python()

import graphviz
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


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

svc = SVC(max_iter=2000, probability = True)
model = svc.fit(X_train, y_train)
model.predict(X_test)

#from sklearn.externals import joblib
#joblib.dump(svc, 'G:/Mi unidad/Desarrollo/desarrolloPython/ML3ERC_svm.pkl')

#twoObs = pd.DataFrame(data=X_test.iloc[0:2,:], columns=df2.columns.values)
#twoObs.to_csv('G:/Mi unidad/Desarrollo/datasets/twoObsERC.csv')
