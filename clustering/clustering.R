# install.packages("kohonen") # 
# install.packages("rworldmap") # 
# load de library 
library("kohonen")
library("rworldmap")

#Carga del fichero con los datos preprocesados
preprocesado = read.csv("~/Dropbox/Uni/Master/DSI/DSI/clustering/datos/preprocesado.csv")

#Selección de las columnas que se usarán en el clustering
clustering.columns=c("Población..INE.","Renta.imponible.media..por.declarante.",
                     "Índice.de.Atkinson.0.5",
                     "Porcentaje.de.Poblacion.Declarante")
preprocesado.num = preprocesado[clustering.columns]

#Adaptación de los datos para el algoritmo
#Se pasa la población a una escala logarítmica para que las grandes ciudades no tengan tanto peso
preprocesado.num["Población..INE."]=log10(preprocesado.num["Población..INE."])
##normalización de los datos usando un escaado min-max
m=sapply(preprocesado.num, min)
M=sapply(preprocesado.num, max)
preprocesado.sc = scale(preprocesado.num,m,M-m)

# Semilla para la inicialización pseudoaleatoria
set.seed(7)
# Entrenamiento
preprocesado.som <- supersom(data = preprocesado.sc, grid = somgrid(4,6,"hexagonal",toroidal = FALSE))
# Gráfico que muestra la influencia de cada columna en cada grupo
plot(preprocesado.som, main = "Data Data")
# La cantidad de elementos en cada neurona
plot(preprocesado.som, type="count")
# La distancia de la cada neurona con sus elementos
plot(preprocesado.som, type="quality")

#Se guardan los grupos es una variable más cómoda de usar
preprocesado.cls=preprocesado.som$unit.classif

#Resumen
preprocesado.summ=aggregate(preprocesado[4:24],list(preprocesado.cls),mean)
#Grupos interesantes:
##Máxima renta imponible media, mucha desigualdad, alto porcentaje de declarantes y poca población
##Máxima población: Ciudades grandes
##Alto porcentaje de declarantes y poco de todos lo demás
##Alto Atkinson y poca renta

#Función para mostrar sobre el mapa de España grupos de municipios con diferentes colores
mostrar.municipios.agrupados.mapa<-function(grupos, nombre){
  #Se crea un selector de los municipios pertenecientes a todos los grupos especificados
  indicesGD=preprocesado.cls == grupos[1]
  for(i in grupos[-1]){
    indicesGD=indicesGD | preprocesado.cls==i
  }
  #Se guardan los grupos seleccionados
  gruposDestacados=preprocesado[indicesGD,]
  gruposDestacados$cls=preprocesado.cls[indicesGD]
  #Se grafica el mapa
  newmap <- getMap(resolution = "low")
  plot(newmap, xlim = c(min(gruposDestacados$LONGITUD_ETRS89), max(gruposDestacados$LONGITUD_ETRS89)), 
       ylim = c(min(gruposDestacados$LATITUD_ETRS89), max(gruposDestacados$LATITUD_ETRS89)),
       asp = 1, main = nombre)
  #Se grafican los municipios
  points(gruposDestacados$LONGITUD_ETRS89, gruposDestacados$LATITUD_ETRS89,
         col=gruposDestacados$cls,pch=gruposDestacados$cls/8)
  #Se grafica la leyenda
  legend(x="topleft", legend = grupos, pch=grupos/8, col=grupos)
}

mostrar.municipios.agrupados.mapa(c(3,7,12,16), "División geográfica entre grupos de municipios")
mostrar.municipios.agrupados.mapa(c(24,23,14), "Municipios habitantes más pobres")
mostrar.municipios.agrupados.mapa(c(1,2,5,6), "Municipios con habitantes más ricos")
