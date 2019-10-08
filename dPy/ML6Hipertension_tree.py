import graphviz
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

def graficar_arbol(grafico = None):
    grafico.format = "png"
    archivo = grafico.render()
    img = mpimg.imshow(archivo)
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close()

df2 = pd.read_csv('G:/Mi Unidad/Desarrollo/datasets/hipertensionCompleta.tsv', delimiter = '\t')
df2 =  df2[(df2["HYP466"] == 1) | (df2["HYP466"] == 2)]

df2.rename(columns={'HYP466':'Dx'},inplace=True)
df2.rename(columns={'HYP70':'familyIncome'},inplace=True)
df2.rename(columns={'HYP143':'familyStructure'},inplace=True)
df2.rename(columns={'HYP145':'healthStatus'},inplace=True)
df2.rename(columns={'HYP526':'heartTrouble'},inplace=True)
df2.rename(columns={'HYP518':'height'},inplace=True)
df2.rename(columns={'HYP520':'howConsiderWeight'},inplace=True)
df2.rename(columns={'HYP175':'incomeHeadFamily'},inplace=True)
df2.rename(columns={'HYP127':'lastDrVisit'},inplace=True)
df2.rename(columns={'HYP60':'maritalStatus'},inplace=True)
df2.rename(columns={'HYP61':'education'},inplace=True)
df2.rename(columns={'HYP50':'race'},inplace=True)
df2.rename(columns={'HYP52':'sex'},inplace=True)
df2.rename(columns={'HYP496':'lastBloodPressure'},inplace=True)

df2 = df2[['Dx','familyIncome','familyStructure','healthStatus','heartTrouble','height','howConsiderWeight',
           'incomeHeadFamily','lastDrVisit','maritalStatus','education','race','sex','lastBloodPressure']]

df2[['lastBloodPressure']] = df2[['lastBloodPressure']].replace(to_replace={'--':1})

X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:,1:14], df2['Dx'], test_size = 0.33, random_state=44, stratify= df2['Dx'])
instancia_arbol = DecisionTreeClassifier(random_state = 0, min_samples_leaf = 30)
instancia_arbol.fit(X_train, y_train)
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))
