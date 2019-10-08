import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


diabetes = pd.read_csv("C:/Users/Liber Proyectos/Downloads/diabetes.csv")
diabetes = diabetes.dropna(thresh=9)
diabetesX = diabetes
diabetesY = diabetes['Outcome']
diabetesX.drop('Outcome',axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(diabetesX, diabetesY, train_size = 0.75, random_state = 0)

clf=SVC(kernel='linear', probability = True)
model = clf.fit(X_train, y_train)
print("Precisión en Testing: {:.3f}".format(model.score(X_test, y_test)))

from sklearn.externals import joblib
joblib.dump(clf, 'G:/Mi unidad/Desarrollo/desarrolloPython/svm2Diabetes.pkl')

twoObs = pd.DataFrame(data=X_test.iloc[0:2,:], columns=diabetesX.columns.values)
twoObs.to_csv('G:/Mi unidad/Desarrollo/datasets/twoObsDiabetes.csv')

model = joblib.load('G:/Mi unidad/Desarrollo/desarrolloPython/svm2Diabetes.pkl')
newObs = pd.read_csv('G:/Mi unidad/Desarrollo/datasets/twoObsDiabetes.csv')
model.predict(newObs.iloc[:,1:10],)
model.predict_proba(newObs.iloc[:,1:10],)

