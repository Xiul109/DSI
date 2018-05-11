# Predicción
Este trabajo consiste en participar en la competición [Dengai](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/) de DrivenData que consiste en intentar predecir en número de casos de dengue que habrá una semana en función de datos atmosféricos.

Estos son los scripts y ficheros que se han utilizado para realizar varias pruebas:
  * **funciones.py:** En este fichero se incluyen varias funciones que se usan desde otros scripts con la finalidad de hacer más ligeros estos últimos.
  * **validacion.py:** En este script se entrenan varios modelos y se prueba su eficacia mediante validación cruzada.
  * **prediccion.py:** En este script se entrena un modelo y se predicen los casos de dengue a partir de las features proporcionadas y genera el fichero correspondiente.
  * **LSTM.py:** Este script utiliza un tipo de red neuronal recurrente (LSTM) para intentar predecir los casos de dengue.

Para obtener más información de todo el proceso, se recomienda leer la memoria ubicada en la carpeta latex/.
