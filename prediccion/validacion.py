import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
from funciones import *
from scipy.signal import medfilt

#Campos seleccionados
select_fields=['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c',
                 'weekofyear']
features_iq, labels_iq,features_sj, labels_sj=cargar_datos_entrenamiento(
                    select_fields=select_fields)

#Se crea un scaler, que hace una normalización basada en zScores
normalFeatures_iq=normalizar_features(pre.StandardScaler(), features_iq,fit=True)
normalFeatures_sj=normalizar_features(pre.StandardScaler(), features_sj,fit=True)

_,nCols=normalFeatures_iq.shape #Número de features a usar por fila

epochs=1500
resultados_iq, resultados_sj=[],[]
for nLayers in [2,5,10]:#El número de capas ocultas a probar en cada iteración
    for f in [lambda x,y,z: embudo(2*x,y,z,2*x),#Las funciones para generar
              lambda x,y,z:uniforme(x,y,z,2*x)]:#topologías a probar
        print("\nCapas ocultas:",nLayers)
        print("Función:",f.__name__)
        #Se definen los parámetros una función para generar el modelo
        modelo=lambda :buildModel(nCols,hiddenLayers=nLayers, fUnits=f,lr=0.05,decay=0.005)
        #Se realiza la validación cruzada para cada una de las ciudades
        print("Validación cruzada Iquitos")
        resultado_iq=validacion_cruzada(features_iq,labels_iq, modelo,
                                        epochs=epochs)
        resultados_iq.append(resultado_iq)
        print("Validación cruzada SanJuan")
        resultado_sj=validacion_cruzada(features_sj,labels_sj, modelo,
                                        epochs=epochs)
        resultados_sj.append(resultado_sj)