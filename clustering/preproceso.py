import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre

#El archivo de entrada tiene que ser descargado desde 
# http://www.fedea.net/renta/docs/datos_renta_municipal.zip
inputRenta="datos/renta2007.xlsx"
inputCoords="datos/CoordenadasMunicipios.csv"
outputFile="datos/preprocesado.csv"

#Se carga el archivo
data=pd.read_excel(inputRenta)
#Se adecua el dataframe, limpiando las primeras filas y cambiando el nombre
data.columns=data.ix[[7]].values.tolist()[0]
data=data[8:]
#Se tiran estas columnas, ya que la información que aportan no es relevante
data=data.drop(columns=["Código INE municipio (5 dígitos)"
                        ,"Código INE Provincia "
                        ,"Código INE Comunidad Autónoma "])
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

#Transforma los campos numéricos susceptibles a ser usados en el análisis
#desde string a float
data=data.astype({key:float for key in numericFields})

#Se cambia el tipo de código INE a int para luego poder juntarla mejor con el
#fichero de las coordenadas
data=data.astype({"Código INE municipio":int})

#Abrir e archivo con las coordenadas de los pueblos y ciudades de España
#El fichero lo he generado a partir de Nomenclator Geográfico de Municipios y Entidades de Población 
#obtenido en http://centrodedescargas.cnig.es/CentroDescargas/equipamiento.do?method=descargarEquipamiento&codEquip=8
coords=pd.read_csv(inputCoords)
coords.COD_INE_CAPITAL=coords.COD_INE_CAPITAL//1000000
coords=coords[["COD_INE_CAPITAL","LATITUD_ETRS89","LONGITUD_ETRS89"]]
data=pd.merge(data,coords,left_on="Código INE municipio",right_on="COD_INE_CAPITAL")
data.loc[data.COD_INE_CAPITAL==39085,"LONGITUD_ETRS89"]/=1000 #Esta coordenada estaba mal puesta

data=data.drop(columns="Código INE municipio")


#función para graficar las correlaciones
def printCorrelations(data, fieldNames, fileName):
    correlaciones=data[fieldNames].corr().values
    
    plt.clf()
    f, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    sns.heatmap(correlaciones, cmap=cmap, vmin=-1,vmax=1,
                square=True, xticklabels=fieldNames,
                yticklabels=fieldNames, linewidths=.5,
                cbar_kws={"shrink": .5}, ax=ax)
    
    fileName="fig/"+fileName
    plt.savefig(fileName, dpi=300, bbox_inches='tight')

print("Usando todas las columnas numéricas")
printCorrelations(data, numericFields, "Correlaciones1.pdf")
#Conclusiones extraídas de las correlaciones
#Índices muy correlacionados y quintiles muy correlacionados con los índices
#Renta agregada no muy relevante en comparación a la media
#Población, población declarante y muestras muy relacionadas, escoger una
#Posible campo nuevo porcentaje de declarantes=declarantes/población
#Cargarse número de observaciones muestrales y población declarante

data["Porcentaje de Poblacion Declarante"]=data["Población declarante (IRPF)"]/data["Población (INE)"]
colsToIgnore=["Índice de Gini ",
            "Renta imponible agregada (IRPF)",
            "Población declarante (IRPF)",
            "Renta imponible media (por habitante)",
            "Número de observaciones muestrales",
            "Renta mediana",
            "Top 1%",
            "Top 0,5%",
            "Top 0,1%",
            "Quintil 1",
            "Quintil 2",
            "Quintil 3",
            "Quintil 4",
            "Quintil 5"]

selectedFields=numericFields[:]
for el in colsToIgnore:
    selectedFields.remove(el)
selectedFields.append("Porcentaje de Poblacion Declarante")

print("Tras preprocesar los datos")
printCorrelations(data,selectedFields, "Correlaciones2.pdf")

#Se aplica PCA para representar los datos en dos dimensiones
from sklearn.decomposition import PCA

min_max_scaler=pre.MinMaxScaler()

auxDF=data[selectedFields]
X=min_max_scaler.fit_transform(auxDF)

pca=PCA(2)

X=pca.fit_transform(X)

print("PCA N=2")
print("Ratio de varianza explicado por varible",pca.explained_variance_ratio_)
print("Ratio de varianza explicado total", np.sum(pca.explained_variance_ratio_))

plt.clf()
plt.scatter(X[:,0],X[:,1])
plt.savefig("fig/pca.pdf", dpi=300, bbox_inches='tight')

#Se guardan los datos preprocesados en el fichero especificado
with open(outputFile,"w") as file:
    data.to_csv(file,sep=",",index=False)