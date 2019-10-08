import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

res = pd.read_csv("G:/Mi unidad/Desarrollo/datasets/ehresp_2014_paraObesidad.csv")
res["obese"] = 0
res = res[res['eusoda']>0]
genFeaturesList = ['eusoda', 'eusnap', 'euincome2', 'eugenhth', 'erincome', 'eudietsoda',
                  'euffyday', 'eufdsit', 'eufastfdfrq','ertseat', 'eudrink', 'eueat',
                  'euexfreq', 'euexercise', 'eufastfd', 'eugenhth', 'eumeat', 'eumilk', 'eustores', 
                  'eustreason', 'euwic']
for feature in genFeaturesList:
    oldSize = len(res)
    newSize = len(res[res[feature]>-1])
    
res["bmi"] = res["erbmi"].astype(float)
res.loc[res['bmi']>30, 'obese'] = 1
target = res["obese"]
resObese =  res[res["obese"] == 1]
resNotObese = res[res["obese"] == 0]

obeseMsk= np.random.rand(len(resObese)) < .7
notObeseMsk = np.random.rand(len(resNotObese)) < .7
trainObese = resObese[obeseMsk]
trainNotObese = resNotObese[notObeseMsk]
testObese = resObese[~(obeseMsk)]
testNotObese = resNotObese[~(notObeseMsk)]

test = testObese.append(testNotObese)
train = trainObese.append(trainNotObese)

trainOriginal = train.copy()
testOriginal = test.copy()
genFeaturesList = ['eusoda', 'eusnap', 'euincome2', 'eugenhth', 'erincome', 'eudietsoda',
                  'euffyday', 'eufdsit', 'eufastfdfrq','ertseat', 'eudrink', 'eueat',
                  'euexfreq', 'euexercise', 'eufastfd', 'eugenhth', 'eumeat', 'eumilk', 'eustores', 
                  'eustreason', 'euwic']
for feature in genFeaturesList:
    train = train[train[feature]>-1]
    trainTarget = train['obese']
    test = test[test[feature]>-1]
    testTarget = test['obese']

topFeaturesList = ['euexfreq', 'eustreason', 'eugenhth', 'ertseat', 'eufastfdfrq']
for feature in topFeaturesList:
  train = train[train[feature]>-1]
  trainTarget = train['obese']
  test = test[test[feature]>-1]
  testTarget = test['obese']

topFeatures = train[topFeaturesList].values
topForest = RandomForestClassifier(max_depth=8)
topForest = topForest.fit(topFeatures, trainTarget)
print(topForest.score(topFeatures, trainTarget))

topForestFeatureImportances = topForest.feature_importances_
print(topForest.feature_importances_)

testTopFeatures = test[topFeaturesList].values
topForest.predict(testTopFeatures)
print(topForest.score(testTopFeatures, testTarget))

importance = topForest.feature_importances_.tolist()
feature_series = pd.Series(data=importance,index=topFeaturesList)

'''
twoObsObesidad = pd.DataFrame(data=testTopFeatures[7:9], columns=topFeaturesList)
twoObsObesidad.to_csv('G:/Mi unidad/Desarrollo/datasets/twoObsObesidad.csv')

from sklearn.externals import joblib
joblib.dump(instancia_red, 'G:/Mi unidad/Desarrollo/desarrolloPython/ML1Obesidad_RF.pkl')

df = pd.read_csv('G:/Mi unidad/Desarrollo/datasets/twoObsObesidad.csv')
randomForestModel = joblib.load('G:/Mi unidad/Desarrollo/desarrolloPython/ML1Obesidad_RF.pkl')


randomForestModel.predict(df.iloc[:,1:6])
randomForestModel.predict_proba(df.iloc[:,1:6])
'''
