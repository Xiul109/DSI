from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Funciones para definir la evolución de la topología de la red neuronal

#Con esta todas las capas tienen el mismo tamaño
def uniforme(nCols,hiddenLayers,i,base=None):
    if not base:
        base=nCols
    return base

#Con esta cada capa va reduciendo su tamaño en función del total de capas
def embudo(nCols,hiddenLayers,i, base=None):
    if not base:
        base=nCols
    return base-i*(nCols//(hiddenLayers))

#Esta función permite definir un modelo de red neuronal
def buildModel(nCols,hiddenLayers=1,fUnits=embudo,lr=0.005,decay=0.01,
               loss="mae"):
    model = Sequential([InputLayer(input_shape=(nCols,))]+
            [Dense(fUnits(nCols,hiddenLayers,i), activation='relu')
                    for i in range(hiddenLayers)]+
            [Dense(1, activation="relu")])
    
    model.compile(optimizer=Adam(lr=lr,decay=decay),loss=loss, metrics=['mean_absolute_error'])
    
    return model

#Esta función entrena modelos y hace una valoración cruzada de los mismos
def validacion_cruzada(features,labels, buildModel, splits=5,epochs=1000, verbose=0): 
    size=features.shape[0]
    regressor=KerasRegressor(buildModel,epochs=epochs, batch_size=size, verbose=verbose)
    kfold = TimeSeriesSplit(n_splits=splits)
    results = cross_val_score(regressor, features, labels, cv=kfold)
    print("Resultados: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    return results

#Esta función entrena un regresor y lo devuelve
def entrenar_regresor(features,labels, buildModel,epochs=1000, verbose=0, plot=False):
    size=features.shape[0]
    regressor=KerasRegressor(buildModel,epochs=epochs, batch_size=size, verbose=verbose)
    history=regressor.fit(features,labels)
    if plot:
        plt.plot(history.history['mean_absolute_error'])
    return regressor

#Predice los casos a partir de un regresor y de unas características
def predecir_casos(regresor, features,scaler=None):
    if scaler:
        features=normalizar_features(scaler,features)
    
    prediccion=np.array(np.round(regresor.predict(features)),np.uint32)

    return prediccion

index_col=[0,1,2]
index_names=["city","year","weekofyear"]
#Esta función carga cualquier fichero de features(ya sea el de entrenamiento
# o el de test) y permite elegir qué campos se seleccionan o cuales se eliminan
def cargar_fichero_features(path,drop_fields=[], select_fields=None):
    #Se carga el fichero con las features
    features=pd.read_csv(path,index_col=index_col)
    #Se eliminan las columnas especificadas
    features=features.drop(columns=drop_fields)
    #Se seleccionan los campos especificados
    if select_fields:
        # Si el campo seleccionado está entre los índices hay que hacer lo
        # siguiente para incluirlo como dato
        for name in index_names:
            if name in select_fields:
                features[name]=features.index.get_level_values(name)
        features=features[select_fields]
    #Se interpolan los valores que faltan
    features=features.interpolate()
    return  features.loc["iq"], features.loc["sj"]

#Esta función carga los datos de entrenamiento utilizando la función de arriba  
def cargar_datos_entrenamiento(drop_fields=[], select_fields=None):
    features_iq,features_sj=cargar_fichero_features(
            "datos/dengue_features_train.csv", drop_fields,select_fields)
    #Se carga el campo de las labels
    labels=pd.read_csv("datos/dengue_labels_train.csv",index_col=index_col)
    return (features_iq, labels.loc["iq"],
            features_sj, labels.loc["sj"])

#Esta función carga los datos de test
def cargar_datos_test(drop_fields=[], select_fields=None):
    return cargar_fichero_features("datos/dengue_features_test.csv",drop_fields, select_fields)

# Esta función normaliza los datos de entrada con el scaler especificado y
# permite elegir si se "entrenar" ese scaler. El scaler puede ser cualquier
# objeto que tenga la misma interfaz que StandardScaler de scikit-learn
def normalizar_features(scaler, features,fit=False):
    if fit:
        scaler.fit(features)
    #Se utiliza ese scaler para normalizar la entrada
    return scaler.transform(features)

# Estas funciones se utilizan para suavizar la entrada de los datos teniendo en
# en cuenta los datos anteriores
def exponential_smoothing(data,alpha=0.5):
    prevRow=data[0]
    auxFeatures=[]
    for row in data:
        auxFeatures.append(prevRow*(1-alpha)+row*alpha)
    return np.array(auxFeatures)

def mean_smoothing(data,window_size=2):
    window=[data[0] for _ in range(window_size)]
    auxData=[]
    for row in data:
        window.pop(0)
        window.append(row)
        auxData.append(np.mean(window,axis=0))
    return np.array(auxData)