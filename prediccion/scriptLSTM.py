import pandas as pd
import numpy as np
import sklearn.preprocessing as pre

#Se carga el fichero con las features
features=pd.read_csv("datos/dengue_features_train.csv")
#Se elimina la columna con la fecha de inicio de la semana
features=features.drop(columns=["week_start_date"])
#Se interpolan los valores que faltan
features=features.interpolate()

#Se crea una copia de los datos para normalizarlos
normalFeatures=features[:]
#Se elimina la columna "city", ya que se pretende entrenar un modelo por ciudad
#,por lo que siempre tomará el mismo valor y no aportará información
normalFeatures=normalFeatures.drop(columns=["city"])
#Se crea un scaler, que hace una normalización basada en zScores
scaler=pre.StandardScaler().fit(normalFeatures)
#Se utiliza ese scaler para normalizar la entrada
normalFeatures=scaler.transform(normalFeatures)

#Se carga el campo de las labels
labels=pd.read_csv("datos/dengue_labels_train.csv")['total_cases']

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from keras.optimizers import Adam

_,nCols=normalFeatures.shape #Número de features a usar por fila
hiddenLayers=1 #Número de capas ocultas a usar
lookback=8#Tamaño de la ventana de tiempo a considerar

def buildModel():
    model = Sequential([LSTM(nCols*2, input_shape=(lookback,nCols), activation='relu')]+
            [Dense(nCols*2-i*(nCols//(hiddenLayers+1)), activation='relu') for i in range(hiddenLayers)]+
            [Dense(1, activation="relu")])
    
    model.compile(optimizer=Adam(lr=0.005,decay=0.01),loss="mean_absolute_error", metrics=['mean_absolute_error'])
    
    return model

#Esta función entrena modelos para hacer una valoración cruzada de los mismos
def validacion_cruzada(f,l, size):   
    regressor=KerasRegressor(buildModel,epochs=1000, batch_size=size, verbose=0)
    kfold = KFold(n_splits=5, random_state=109)
    results = cross_val_score(regressor, f, l, cv=kfold)
    print("Resultados: %.2f (%.2f) MAE" % (results.mean(), results.std()))
    return results

#Esta función entrena un regresor y lo devuelve
def entrenar_regresor(f,l,size):
    regressor=KerasRegressor(buildModel,epochs=1000, batch_size=size, verbose=0)
    regressor.fit(f,l)
    return regressor

def transformar(f):
    auxSample=[np.zeros(nCols) for _ in range(lookback)]
    new_features=[]
    for row in f:
        auxSample.pop(0)
        auxSample.append(row)
        new_features.append(np.array(auxSample))
    
    return np.array(new_features)

#Esta selecciona los datos de una ciudad y le aplica las dos funciones anteriores
def validar_y_entrenar(city):
    city_labels=np.array(labels[features["city"]==city])
    
    city_features=np.array(normalFeatures[features["city"]==city])
    city_features=transformar(city_features)
    size=city_features.shape[0]
    
    resultados=validacion_cruzada(city_features,city_labels,size)
    regresor=entrenar_regresor(city_features,city_labels,size)
    
    return resultados,regresor

#Se prueba y entrenan los datos de cada ciudad
print("Validación cruzada Iquitos")
resultados_iq,regresor_iq=validar_y_entrenar("iq")
print("Validación cruzada SanJuan")
resultados_sj,regresor_sj=validar_y_entrenar("sj")

#Se cargan los datos de prueba
features_test=pd.read_csv("datos/dengue_features_test.csv")
city_test=features_test["city"]
features_test=features_test.drop(columns=["week_start_date"])
features_test=features_test.interpolate()

#Predice los casos para una ciudad concreta y con un regresor específico
def predecir_casos(regresor, city):
    features_test_city=features_test[city_test==city]
    features_test_city_transformada=transformar(scaler.transform(features_test_city.iloc[:,1:]))
    
    prediccion=np.array(np.round(regresor.predict(features_test_city_transformada)),int)
    
    labels_test_city=features_test_city[["city","year","weekofyear"]]
    labels_test_city["total_cases"]=prediccion
    return labels_test_city

#Se predicen los casos para las dos ciudades y se unen en un solo dataframe
labels_test_iq=predecir_casos(regresor_iq,"iq")
labels_test_sj=predecir_casos(regresor_sj,"sj")
labels_test=labels_test_sj.append(labels_test_iq)
#Se genera un dataframe con los campos requeridos para poder subirlo a drivendata
labels_test.to_csv("datos/dengue_labels_test.csv",index=False)