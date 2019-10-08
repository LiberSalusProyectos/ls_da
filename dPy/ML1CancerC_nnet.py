# de https://www.kaggle.com/randyrose2017/for-beginners-using-keras-to-build-models

# para traer el interprete de python en RStudio
# reticulate::repl_python()

import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('darkgrid')
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_full = pd.read_csv('C:/Users/Liber Proyectos/Downloads/kag_risk_factors_cervical_cancer.csv')
df_full.info()
df_fullna = df_full.replace('?', np.nan)
df_fullna.isnull().sum()
df = df_fullna
df = df.convert_objects(convert_numeric=True)
df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())
df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
df['Smokes'] = df['Smokes'].fillna(1)
df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())
df['IUD'] = df['IUD'].fillna(0) # Under suggestion
df['IUD (years)'] = df['IUD (years)'].fillna(0) #Under suggestion
df['STDs'] = df['STDs'].fillna(1)
df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())
df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].median())
df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].median())
df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].median())
df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())
df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].median())
df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())
df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].median())
df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())
df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())
df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())
df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())
df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(df['STDs: Time since first diagnosis'].median())
df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].median())
df['Smokes'] = df['Smokes'].astype('category')
df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].astype('category')
df['IUD'] = df['IUD'].astype('category')
df['STDs'] = df['STDs'].astype('category')
df['Dx:Cancer'] = df['Dx:Cancer'].astype('category')
df['Dx:CIN'] = df['Dx:CIN'].astype('category')
df['Dx:HPV'] = df['Dx:HPV'].astype('category')
df['Dx'] = df['Dx'].astype('category')
df['Hinselmann'] = df['Hinselmann'].astype('category')
df['Citology'] = df['Citology'].astype('category')
df['Schiller'] = df['Schiller'].astype('category')
#df = pd.get_dummies(data=df, columns=['Smokes','Hormonal Contraceptives','IUD','STDs','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller'])
#df.isnull().sum()
df_data = df
np.random.seed(42)
df_data_shuffle = df_data.iloc[np.random.permutation(len(df_data))]
df_train = df_data_shuffle.iloc[1:686, :]
df_test = df_data_shuffle.iloc[686: , :]

#df_train_feature = df_train[['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)','Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)','STDs:condylomatosis', 'STDs:cervical condylomatosis','STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis','STDs:syphilis', 'STDs:pelvic inflammatory disease','STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS','STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','STDs: Time since first diagnosis', 'STDs: Time since last diagnosis','Smokes_0.0', 'Smokes_1.0','Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0','IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1','Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1','Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1','Schiller_0','Schiller_1']]
df_train_feature = df_train.iloc[:,:35]
train_label = np.array(df_train['Biopsy'])
#df_test_feature = df_test[['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)','Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)','STDs:condylomatosis', 'STDs:cervical condylomatosis','STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis','STDs:syphilis', 'STDs:pelvic inflammatory disease','STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS','STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','STDs: Time since first diagnosis', 'STDs: Time since last diagnosis','Smokes_0.0', 'Smokes_1.0','Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0','IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1','Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1','Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1','Schiller_0','Schiller_1']]
df_test_feature = df_test.iloc[:,:35]
test_label = np.array(df_test['Biopsy'])
#from sklearn import preprocessing
#minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
#train_feature = minmax_scale.fit_transform(df_train_feature)
train_feature = df_train_feature
#test_feature = minmax_scale.fit_transform(df_test_feature)
test_feature = df_test_feature
train_feature.shape

# de Mi Unidad/Desarrollo/desarrolloPython/nnet1_2texto.py
from sklearn.neural_network import MLPClassifier
instancia_red = MLPClassifier(solver='lbfgs', random_state=0)
instancia_red.fit(train_feature, train_label)
pred = instancia_red.predict(test_feature,)
unique_items, counts = np.unique(pred, return_counts=True)
counts
print("Precisión en Testing: {:.3f}".format(instancia_red.score(test_feature, test_label)))

# guardar modelo (ref https://www.youtube.com/watch?v=5X3xWlJ2Ozw)
#import joblib
#joblib.dump(instancia_red, 'G:/Mi Unidad/Desarrollo/desarrolloPython/ANN1CancerC.pkl')

# guardar un par de observaciones para validar el modelo guardado
#tmp = test_feature.iloc[7:9,:]
#twoObsCancerC = pd.DataFrame(data=tmp, columns=df_test_feature.columns.values)
#twoObsCancerC.to_csv('G:/Mi Unidad/Desarrollo/datasets/twoObsCancerC.csv')

# se carga el modelo
#df_full = pd.read_csv('G:/Mi unidad/Desarrollo/datasets/twoObsCancerC.csv')
#instancia_red = joblib.load('G:/Mi unidad/Desarrollo/desarrolloPython/ANN1CancerC.pkl')
#format(instancia_red.predict(df_full.iloc[:,1:47],))
#instancia_red.predict_proba(df_full.iloc[:,1:],)

#tmp = pd.DataFrame(data=train_feature, columns=df_train_feature.columns.values)
#tmp.to_csv('G:/Mi unidad/Desarrollo/datasets/trainXCancerC.csv')
