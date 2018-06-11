import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

#NOTA: A veces la ejecución falla, hasta donde he investigado se ve que es por
#un bug de pandas. Si se ejecuta de nuevo no suele fallar.

def cargarFichero(path, pAlumnos=0.5):
    """
    Argumentos
    ----------
    path: str
        Ruta del archivo a cargar con un fichero de entrenamiento. Sólo se han
        hecho pruebas con el fichero de 2005-2006
    pAlumnos: float
        Porcentaje de alumnos que se quiere utilizar
    
    Return
    ------
    DataFrame con los datos del fichero
    """
    #Se lee el fichero especificando que el índice es la primera columna y que
    #el separador es un tabulador
    fichero=pd.read_csv(path,sep="\t", index_col=0)
    
    #Para hacer una selección de alumnos, primero se obtiene una lista con
    #todos los alumnos en los que solo aparezca una vez cada uno
    alumnos=fichero["Anon Student Id"].drop_duplicates()
    #A continuación se seleccionan aquellos en los cuales un número generado
    #aleatoriamente sea menor que pAlumnos
    alumnosSeleccionados=alumnos[np.random.random(len(alumnos))<pAlumnos]
    #Por último se seleccionan aquellas filas del fichero que incluyen a los
    #alumnos seleccionados
    fichero=fichero[fichero["Anon Student Id"].isin(alumnosSeleccionados)]
    
    #Se transforma a pandas.Datetime las fechas que aparecen en el fichero
    for campo in ["Step Start Time","Step End Time","First Transaction Time",
                  "Correct Transaction Time"]:
        fichero[campo]=pd.to_datetime(fichero[campo])
    
    #Tanto KC como opportunity se separan por la secuencia '~~' y se convierten
    #en listas. En el caso de la opportunity además se tiene en transforma para 
    #que las listas sean de entero y no de cadenas de texto
    fichero["KC(Default)"]=fichero["KC(Default)"].str.split("~~")
    fichero["Opportunity(Default)"]=fichero["Opportunity(Default)"].str.split(
            "~~").apply(lambda x:list(map(int,x)) if type(x) is list else x)
    
    #Por último se sustituyen los valores nulos por listas vacías
    for row in fichero.loc[fichero["KC(Default)"].isnull()].index:
        fichero.at[row,"KC(Default)"]=[]
        fichero.at[row,"Opportunity(Default)"]=[]
    
    return fichero

def agruparStepsEnProblemas(steps):
    """
    Agrupa los pasos en problemas, con los cuales se operará en este script.
    Además añade algunos campos adicionales y renombra algunos existentes.
    
    Argumentos
    ----------
    steps: DataFrame
        Fichero tal cual lo devuelve la función cargarFichero.
    Return
    ------
    DataFrame con los pasos agrupados en problemas
    """
    #Se agrupan los pasos mediante los campos "Anon Student Id", "Problem Name"
    #"Problem Hierarchy","Problem Name" y "Problem View"; que son los que
    #identifican de manera única cada vez que un problema está siendo resuelto
    #por un alumno
    gruposProblemas=steps.groupby(["Anon Student Id", "Problem Hierarchy",
                               "Problem Name","Problem View"])
    
    #Se crean los siguientes campos para cada problema: suma de fallos, pistas
    #y aciertos por paso; media de Correct First Attempt, lo cual se puede
    #entender como el porcentaje de pasos que se han completado a la primera
    #para ese problema; el tiempo de inicio(el mínimo de los pasos) y el tiempo
    #de fin(el máximo de los pasos) del problema; la suma de las duraciones de
    #los pasos y la suma de los KC (al ser listas, las sumas concatenan)
    problemas=gruposProblemas[["Incorrects","Hints","Corrects"]].sum().join(
              gruposProblemas["Correct First Attempt"].mean()).join(
              gruposProblemas["Step Start Time"].min()).join(
              gruposProblemas["Step End Time"].max()).join(
              gruposProblemas["Step Duration (sec)"].sum()).join(
              gruposProblemas["KC(Default)"].sum())

    #Se renombran las columnas para trabajar mejor con ellas
    problemas.rename(columns={"Step Start Time"    :"Problem_start_time",
                              "Step End Time"      :"Problem_end_time",
                              "KC(Default)"        :"KCs",
                              "Step Duration (sec)":"Step_duration_sum"},
                        inplace=True)
    
    #Se añade el campo "Problem_duration", que se calcula como el tiempo final
    #menos el tiempo de inicio del problema. Este campo se crea porque la suma
    #de las duraciones de los pasos no siempre da de resultado la resta del
    #tiempo de inicio menos el tiempo de fin. Esto se debe a que algunos campos
    #faltan y a que el alumno puede realizar varios pasos simultáneamente.
    problemas["Problem_duration"]=(problemas["Problem_end_time"]-\
                                   problemas["Problem_start_time"]).dt.seconds
    #Si no se ha podido calcular alguna fila del campo anterior, esta se
    #completa con el valor de la suma de duraciones de los pasos.
    problemas["Problem_duration"].fillna(problemas["Step_duration_sum"],
                                         inplace=True)
    
    #Se crean nuevas columnas, una por cada KC diferente y se inicializan a 0
    for KC in set(problemas.KCs.sum()):
        problemas[KC]=0
    #Para cada problema, si ese problema contiene un determinado KC, entonces
    #se iguala a 1 la intersección entre la fila del problema y la columna
    #correspondiente a ese KC
    for index,row in problemas.iterrows():
        for key in row.KCs:
            problemas.at[index,key]=1
    #Se elimina la columna KCs, tras haberla transformado en varias columnas
    problemas.drop(columns=["KCs"], inplace=True)
    
    #Se aplica el algoritmo KMeans utilizando los campos Incorrects, Hints y
    #Problem_duration para establecer la dificultad asociada a un problema. Se
    #utilizan 3 grupos que representarán a dificultades: fácil, moderada y
    #difícil.
    camposKmeans=problemas[["Incorrects","Hints"]]
    kmeans=KMeans(3)
    problemas["grupo"]=kmeans.fit_predict(camposKmeans)
    #Se reordenan las columnas para que el grupo esté al principio por 
    #comodidad, ya que es la clase que se quiere predecir y al final estarán 
    #las características.
    problemas=problemas[["grupo"]+list(problemas.columns[:-1])]
    
    return problemas
 
def agruparEsquemasDeProblemas(problemas):
    """
    Agrupa los problemas en esquemas definitorios de cada uno, que incluyen la
    cuenta de elementos totales de cada agrupación y valores medios de algunos
    campos (por ejemplo la duración, las pistas o los KCs). Debido a que para
    resolver un problema no hace falta siempre tomar los mismos pasos, es
    habitual que la media de los KCs sea diferente de 1 o 0; lo cual se puede
    interpretar como la importancia que tiene un KC a la hora de resolver un
    problema (1 es que el KC es imprescindible y 0 nada importante).
    
    Argumentos
    ----------
    problemas: DataFrame
        Agrupación de problemas devuelta por función agruparStepsEnProblemas
    
    Return
    ------
    DataFrame con los esquemas de cada problema
    """
    gb=problemas.groupby("Problem Name")
    return  gb.size().to_frame(name="counts").join(
            problemas.drop(columns="grupo").groupby("Problem Name").mean())

def prepararDatosDelModelo(problemas, esquemaProblemas):
    """
    Prepara los datos que serán utilizados para entrenar el modelo, los cuales
    son relativos a los conocimientos del alumno y a los KCs necesarios para
    resolver el problema, además de algunos campos adicionales.
    
    Argumentos
    ----------
    problemas: DataFrame
        Agrupación de problemas devuelta por función agruparStepsEnProblemas.
    esquemaProblemas: DataFrame
        Esquemas generales de cada problema devueltos por la función
        agruparEsquemasDeProblemas.
    
    Return
    ------
    DataFrame con los datos preparados para entrenar el modelo
    """
    #Se guardan en variables:
    # - Las columnas correspondientes a los KC del problema
    columnasKC=problemas.columns[9:]
    # - Las columnas correspondientes al aprendizaje de los KC por parte del
    #   alumno.
    aprendizajeKC="aprendizaje"+columnasKC
    # - Otros campos adicionales que almacenarán la media de varios valores de
    #   cada alumno a través de varios problemas.
    otrosCampos=["t_medio","hints_medio","incorrects_medio","corrects_medio",
                 "cfa_medio","n_problemas_previos"]
    # - Una lista con el nombre de los columnas del DataFrame que hay que
    #   promediar
    camposAPromediar=["Problem_duration","Hints","Incorrects","Corrects",
                      "Correct First Attempt"]
    # - Una lista incluyendo los campos relativos al aprendizaje de KCs y los 
    #   campos que representan las medias de varios valores.
    nuevosCampos=list(aprendizajeKC)+otrosCampos
    
    #Se crea la variable modelData en la cual se almacena el DataFrame
    #problemas con las columnas nuevas añadidas e inicializadas a 0.
    modelData=problemas.assign(**{key:0 for key in nuevosCampos})
    #Ordena los campos en función de la fecha de fin, ya que será necesario
    #tener en cuenta el orden temporal para caracterizar el aprendizaje del
    #alumno.
    modelData.sort_values("Problem_end_time", inplace=True)
    
    #Se guardan en variables el número de KCs diferentes y el número de nuevas
    #columnas.
    nKC=len(aprendizajeKC)
    nCols=len(nuevosCampos)
    
    #Para cada problema se considera que, al completarlo, el alumno aprende los
    #KCs que aparecen en este, sin embargo, este aprendizaje es inversamente 
    #proporcional a las pistas que el alumno recibe (ya que si la herramienta
    #le ha dicho como resolver el problema, este no habrá podido demostrar su
    #conocimiento en el tema) y ha de estar entre 0 y 1, es decir, el
    #porcentaje del KC que el alumno ha aprendido. Con esto en mente se ha
    #calculado una penalización del aprendizaje como 2^(-hints/2), lo cual
    #siempre devolverá un número entre 0 y 1 (ya que hints siempre es positivo 
    #o 0), siendo más cercano a 0 cuanto mayor sea la cantidad de pistas
    #recibida.
    penalizacionHints=2.0**(-modelData.Hints/2)
    penalizacionesKC=modelData.loc[:,columnasKC].mul(penalizacionHints,axis=0)
    
    #Se precalcula para ganar eficiencia, ya que se utilizará varias veces en
    #el bucle
    segundosPorSemana=60*60*24*7
    
    #Índices de los diferentes campos que se utilizarán
    indiceTmedio=len(aprendizajeKC)
    indiceHintsMedio=indiceTmedio+1
    indiceIncorrectsMedio=indiceHintsMedio+1
    indiceCorrectsMedio=indiceIncorrectsMedio+1
    indiceCFAMedio=indiceCorrectsMedio+1
    indiceNProblemasPrevios=indiceCFAMedio+1
    
    #Este bucle itera sobre los problemas agrupados por alumno
    for alumno, df in modelData.groupby(level=0):
        fila1=df.iloc[0]
        #Al inicio el aprendizaje del alumno es el correspondiente a la primera
        #fila
        aprendizajePrevio=penalizacionesKC.loc[df.index[0]]
        #Y la fecha en la que se completó el primer problema
        fechaPrevia=fila1["Problem_end_time"]
        #Se inicializan estas variables que representan la suma de los valores
        #de la duración de los problemas previos
        tSum,hSum,iSum,cSum,cfaSum=fila1[camposAPromediar]
        #Se crea una variable auxiliar que almacenará los resultados parciales
        #de cada alumno. Esto se hace así porque la asignación de valores a los
        #DataFrames es bastante ineficiente y ralentiza de sobre manera el
        #bucle siguiente
        auxData=np.zeros((df.shape[0],nCols))
        #Se itera el bucle de las filas correspondientes a cad alumno y se
        #omite la primera debido a que todos los cálculos dan cero
        for i,(index, row) in enumerate(df.iloc[1:].iterrows()):
            #Segundos pasados desde el fin del problema anterior hasta el fin
            #del siguiente. Se utiliza la fecha de fin en ambos porque hay
            #algunos problemas que comienzan antes de que termine el anterior.
            segundos=(row["Problem_end_time"]-fechaPrevia).seconds
            #El multiplicador del aprendizaje escala de la misma manera que la
            #penalización por pistas. En este caso se considera que el alumno
            #ha olvidado el 50% de un KC pasada una semana.
            multiplicador=2**(-segundos/segundosPorSemana)
            auxData[i,:nKC]=aprendizajePrevio*multiplicador
            #Se calcula n para hacer las medias de los diferentes valores
            n=i+1
            #y se añaden a auxData
            auxData[i,indiceTmedio]=tSum/n
            auxData[i,indiceHintsMedio]=hSum/n
            auxData[i,indiceIncorrectsMedio]=iSum/n
            auxData[i,indiceCorrectsMedio]=cSum/n
            auxData[i,indiceCFAMedio]=cfaSum/n
            auxData[i,indiceNProblemasPrevios]=i
            
            #El aprendizaje previo para el siguiente problema se calcula como 
            #el máximo entre los valores del aprendizaje anterior y el
            #aprendizaje adquirido con el problema actual.
            aprendizajePrevio=np.maximum(auxData[i,:nKC],
                                         penalizacionesKC.loc[index].values)
            #La fecha previa ahora es la fecha de fin del problema actual
            fechaPrevia=row["Problem_end_time"]
            #A cada acumulador se le suma el valor correspondiente asociado a
            #el problema en cuestión
            tSum+=row["Problem_duration"]
            hSum+=row["Hints"]
            iSum+=row["Incorrects"]
            cSum+=row["Corrects"]
            cfaSum+=row["Correct First Attempt"]
        #Se introduce en los datos del modelo los valores calculados en el
        #bucle
        modelData.loc[df.index,nuevosCampos]=auxData
    
    
    slNone=slice(None)
    #Se cambian los KCs específicos del problema por los del esquema del 
    #problema
    for index,row in esquemaProblemas.iterrows():
        modelData.loc[(slNone,slNone,index),columnasKC]=row[columnasKC].values
        
    return modelData

def entrenarYValidarModelo(modelData):
    """
    Define el modelo de aprendizaje (RandomForest), lo entrena, lo valida
    mediante validación cruzada usando las métricas de precisión y F1 y
    devuelve la matriz de confusión.
    
    Argumentos
    ----------
    modelData: DataFrame
        Datos de entrada al modelo devueltos por la función 
        prepararDatosDelModelo.
    Return
    ------
    numpy.ndarray que contiene la mat
    """
    #Número de columnas a usar como features
    nCols=len(modelData.columns[9:])
    #El clasificador a utilizar será un RandomForest y utilizará como
    #número de estimadores 2 por la raiz cuadrada del número de columnas. Los
    #demás parámetros toman los valores por defecto
    clasificador=RandomForestClassifier(n_estimators=int(2*nCols**0.5))
    
    #Se guardan los datos de entrenamiento en la variable train
    train=modelData.iloc[:,9:]
    
    #Se crea un objeto kFold para llevar a cabo la validación cruzada
    kfold=KFold()
    #Primero se evalua la precisión del modelo
    accuracy=cross_val_score(clasificador,train,modelData.grupo,
                          cv=kfold, scoring="accuracy")
    print("Precisión:", accuracy)
    print("Precisión media:",np.mean(accuracy),"(",np.std(accuracy),")")
    
    #Y luego el valor del f1, el cual se calcula por separado para cada label y
    #luego se promedia
    f1=cross_val_score(clasificador,train,modelData.grupo,
                          cv=kfold, scoring="f1_macro")
    print("F1:",f1)
    print("F1 media:",np.mean(f1),"(",np.std(f1),")")
    
    #Se realiza un split de los datos en prueba y entrenamiento
    trainX, testX, trainY, testY=train_test_split(train, modelData.grupo,
                                                  test_size=0.4)
    #Se entrena al modelo
    clasificador.fit(trainX,trainY)
    #Y se predicen los resultados
    predictY=clasificador.predict(testX)
    
    print("Importancia de cada característica:",sorted(zip(
            clasificador.feature_importances_, modelData.columns[9:])))
    
    #Se computa la matriz de confusión
    cm=confusion_matrix(testY,predictY)
    
    return cm

def mostrarYEtiquetarGrupos(problemas):
    """
    Muestra por pantalla los grupos que ha generado el KMeans y los etiqueta en
    función de su dificultad.
    
    Argumentos
    ----------
    problemas: DataFrame
        Agrupación de problemas devuelta por función agruparStepsEnProblemas.
    
    Returns
    -------
    Lista con la etiqueta asociada a cada grupo.
    """
    #Se agrupan los problemas en función del grupo y se calcula la media
    grupos = problemas.groupby("grupo")[["Hints","Incorrects"]].mean()
    #La lista de índices de los grupos
    indices=[0, 1, 2]
    #Un array para guardar las etiquetas de los grupos
    etiquetas=np.empty(3,object)
    #Los índices del grupo con menos y más Hints
    iMax,iMin = grupos["Hints"].idxmax(), grupos["Hints"].idxmin()
    #Se asigna la etiqueta difícil al índice correspondiente
    etiquetas[iMax]="Difícil"
    #Se elimina el índice difícil de la lista de índices
    indices.remove(iMax)
    #Se asigna la etiqueta fácil al índice correspondiente
    etiquetas[iMin]="Fácil"
    #Se elimina el índice fácil de la lista de índices
    indices.remove(iMin)
    #El último índice que queda es el de la dificultad moderada
    etiquetas[indices.pop()]="Moderada"
    
    #Se grafican los grupos
    plt.scatter(problemas["Hints"],problemas["Incorrects"],
                c=problemas["grupo"])
    plt.title("Grupos")
    plt.xlabel("Hints")
    plt.ylabel("Incorrects")
    plt.show()
    
    return etiquetas

def mostrarMatrizDeConfusión(cm,etiquetas):
    """
    Muestra gráficamente una matriz de confusión.
    
    Argumentos
    ----------
    cm: 2D numpy.ndarray
        Matriz de confusión.
    etiquetas: array like
        Etiquetas de cada grupo.
    """
    df_cm = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)
    fig = plt.figure()
    
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="viridis")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
                                 ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
                                 ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.show()
    return fig

#Ejecución de las funciones
print("""AVISO: La ejecución completa puede tardar un tiempo notable 
      (aproximadamente 1:30 min seleccionando un 10% de los alumnos y 8 min
      con el 50% de los alumnos)""")
print("Cargando el fichero...")
datosFichero=cargarFichero("datos/algebra_2005_2006_train.txt")
print("Agrupando steps en problemas...")
problemas=agruparStepsEnProblemas(datosFichero)
print("Mostrando las agrupaciones graficamente...")
etiquetas=mostrarYEtiquetarGrupos(problemas)
print("Creando esquemas definitorios de cada problema a partir de la",
      "agrupación anterior...")
esquemaProblemas=agruparEsquemasDeProblemas(problemas)
print("Generando los datos que se usarán en el modelo...")
modelData=prepararDatosDelModelo(problemas,esquemaProblemas)
print("Entrenando y validando el modelo...")
cm=entrenarYValidarModelo(modelData)
print("Mostrando la matriz de confusión gráficamente...")
mostrarMatrizDeConfusión(cm,etiquetas)


