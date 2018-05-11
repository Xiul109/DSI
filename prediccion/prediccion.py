import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
from funciones import *

#Campos seleccionadoss
select_fields=['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c',
                 'weekofyear']
features_iq, labels_iq, features_sj, labels_sj=cargar_datos_entrenamiento(
                                                select_fields=select_fields)

#Se crea un scaler, que hace una normalización basada en zScores, para cada ciudad
scaler_iq=pre.StandardScaler()
normalFeatures_iq=normalizar_features(scaler_iq, features_iq, fit=True)

scaler_sj=pre.StandardScaler()
normalFeatures_sj=normalizar_features(scaler_sj, features_sj, fit=True)

_,nCols=normalFeatures_sj.shape #Número de features a usar por fila

#Se definen los parámetros una función para generar el modelo
modelo=lambda :buildModel(nCols, hiddenLayers=nCols*2,
                          fUnits=lambda x,y,z: uniforme(x,y,z,2*x),
                          lr=0.05,decay=0.005)

epochs=1500
#Se entrena un modelo para cada ciudad
regresor_iq=entrenar_regresor(normalFeatures_iq,labels_iq, modelo,epochs,verbose=1)
regresor_sj=entrenar_regresor(normalFeatures_sj,labels_sj, modelo,epochs,verbose=1,plot=True)

#Se cargan los datos de prueba
features_test_iq,features_test_sj=cargar_datos_test(select_fields=select_fields)

#Se predicen los casos para las dos ciudades y se unen en un solo dataframe
prediccion_iq=predecir_casos(regresor_iq,features_test_iq,scaler_iq)
prediccion_sj=predecir_casos(regresor_sj,features_test_sj,scaler_sj)

#Se caga el fichero con el formato de envío
submission = pd.read_csv("datos/submission_format.csv",
                         index_col=[0, 1, 2])
#Se concatenan las predicciones para cada ciudad
submission.total_cases=np.concatenate([prediccion_sj, prediccion_iq])
#Se genera un dataframe con los campos requeridos para poder subirlo a drivendata
submission.to_csv("datos/dengue_labels_test.csv")