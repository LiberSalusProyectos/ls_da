import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df2 = pd.read_csv('G:/Mi Unidad/Desarrollo/datasets/hipertension8var.csv')
df2 =  df2[(df2["Dx"] == 1) | (df2["Dx"] == 2)]

df2.rename(columns={'HYP193':'weight'},inplace=True)
df2.rename(columns={'HYP95':'familySize'},inplace=True)
df2.rename(columns={'HYP532':'adviceSmoking'},inplace=True)
df2.rename(columns={'HYP475':'adviceSalt'},inplace=True)
df2.rename(columns={'HYP53':'age'},inplace=True)
df2.rename(columns={'HYP529':'cigarettesPerDay'},inplace=True)
df2.rename(columns={'HYP76':'classWorked'},inplace=True)
df2.rename(columns={'HYP525':'diabetes'},inplace=True)
# posibles a añadir: 
# pocos valores HYP528, HYP527, HYP526, HYP520, HYP498, HYP494
# muchos valores HYP518 (0-99), HYP515, HYP513, HYP511 (0-99), HYP508 (0-999), PSUNUMR (3-999), WEEKCEN (0-28)

X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:,1:9], df2['Dx'], test_size = 0.33, random_state=44, stratify= df2['Dx'])
clf = MLPClassifier(solver='lbfgs', random_state=0)
clf.fit(X_train, y_train)

print("Las predicciones en Testing son :{}".format(clf.predict(X_test,)))
print("PrecisiÃ³n en Testing: {:.3f}".format(clf.score(X_test, y_test)))
