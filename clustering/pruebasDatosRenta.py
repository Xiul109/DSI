import pandas as pd 
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cls
import sklearn.preprocessing as pre

#Se carga el archivo
data=pd.read_excel("renta2007.xlsx")
#Se adecua el dataframe, limpiando las primeras filas y cambiando el nombre
data.columns=data.ix[[7]].values.tolist()[0]
data=data[8:]
#Se tira esta columna, ya que es redundante
data=data.drop(columns="Código INE municipio (5 dígitos)")
#Se quitan las filas con campos vacios(6 en total)
data=data[data["Índice de Gini "] != 'n.d.']
#Se seleccionan los campos numéricos para cambiarles el tipo a float
numericFields=["Número de observaciones muestrales",
               "Población (INE)",
               "Población declarante (IRPF)",
               "Renta imponible agregada (IRPF)",
               "Renta imponible media (por declarante)",
               "Renta imponible media (por habitante)",
               "Renta mediana",
               "Índice de Gini ",
               "Índice de Atkinson 0,5",
               "Top 1%",
               "Top 0,5%",
               "Top 0,1%",
               "Quintil 1",
               "Quintil 2",
               "Quintil 3",
               "Quintil 4",
               "Quintil 5"]

data=data.astype({key:float for key in numericFields})
data=data.astype({"Código INE municipio":int})

#Abrir e archivo con las coordenadas de los pueblos y ciudades de España
#El fichero lo he generado a partir de Nomenclator Geográfico de Municipios y Entidades de Población 
#obtenido en http://centrodedescargas.cnig.es/CentroDescargas/equipamiento.do?method=descargarEquipamiento&codEquip=8
#coords=pd.read_csv("CoordenadasMunicipios.csv")
#coords.COD_INE_CAPITAL=coords.COD_INE_CAPITAL//1000000
#coords=coords[["COD_INE_CAPITAL","LATITUD_ETRS89","LONGITUD_ETRS89"]]
#data=pd.merge(data,coords,left_on="Código INE municipio",right_on="COD_INE_CAPITAL")
#data.loc[data.COD_INE_CAPITAL==39085,"LONGITUD_ETRS89"]/=1000 #Esta coordenada estaba mal puesta


#función para imprimir las correlaciones
def printCorrelations(data, fieldNames):
    correlaciones=data[fieldNames].corr().values
    
    for i in range(len(correlaciones)):
        correlaciones[i,i:]=0
    
    correlaciones=correlaciones[1:,:-1]
    
    plt.clf()
    f, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    sns.heatmap(correlaciones, cmap=cmap, vmax=.8,
                square=True, xticklabels=fieldNames[:-1],
                yticklabels=fieldNames[1:], linewidths=.5,
                cbar_kws={"shrink": .5}, ax=ax)
    plt.show()

print("Using all the numeric columns")
printCorrelations(data, numericFields)
#Conclusiones extraídas de las correlaciones
#Q5 e índices muy correlacionados, escoger solo un índice
#Renta agregada no muy relevante en comparación a la media
#Población, población declarante y muestras muy relacionadas, escoger una
#Posible campo nuevo porcentaje de declarantes=declarantes/población
#Cargarse número de observaciones muestrales y población declarante

data["Porcentaje de Poblacion Declarante"]=data["Población declarante (IRPF)"]/data["Población (INE)"]
colsToDrop=["Índice de Gini ",
            "Renta imponible agregada (IRPF)",
            "Población declarante (IRPF)",
            "Número de observaciones muestrales",
            "Renta mediana",
            "Top 0,5%"]
#data=data.drop(columns=colsToDrop)
newNames=numericFields[:]
for el in colsToDrop:
    newNames.remove(el)
newNames.append("Porcentaje de Poblacion Declarante")
print("Preprocesing done")
printCorrelations(data,newNames)

from sklearn.decomposition import PCA

min_max_scaler=pre.MinMaxScaler()

X=min_max_scaler.fit_transform(data[newNames].values)

pca=PCA(2)

X=pca.fit_transform(X)

print(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_))

plt.scatter(X[:,0],X[:,1])
plt.show()

with open("preprocesado.csv","w") as file:
    data.to_csv(file,sep=",",index=False)