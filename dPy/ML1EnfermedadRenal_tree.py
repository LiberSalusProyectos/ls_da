# de https://www.kaggle.com/mansoordaku/ckdisease/downloads/kidney_disease.csv/1
# descripcion de variables https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease#
#reticulate::repl_python()

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


df = pd.read_csv('C:/Users/Liber Proyectos/Downloads/kidney_disease.csv')

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
y_train.value_counts()

instancia_arbol = DecisionTreeClassifier(random_state = 0, min_samples_leaf = 5)
instancia_arbol.fit(X_train, y_train)

print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))
prediccion = instancia_arbol.predict(X_test)

dot_data = export_graphviz(instancia_arbol, out_file = None, class_names = ['1', '0'],
                           feature_names = list(df2.iloc[:,:-1].columns.values), filled = True)
grafico = graphviz.Source(dot_data)
graficar_arbol(grafico)
