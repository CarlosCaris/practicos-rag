# Proyecto: rag-chile-food-regulations

_El presente proyecto trata acerca de la utilización de un modelo de Generación Aumentada con Recuperación (RAG por sus siglas en inglés), aplicado al contexto de las regulaciones sanitarias alimenticias en Chile._

## Pre-requisitos 
_Para poder trabajar con el repositorio y sus dependencias, necesitas contar con Python en versiones 3 en adelante._


## Pasos para el despliegue:

### 1. Clonación de repositorio

El paso inicial es clonar el repositorio en su entorno local. Para esto ejecute el siguiente comando:

```
git clone https://github.com/fernando-echiburug/practicos-rag.git
```

Una vez clonado, entre a la carpeta del repositorio (por defecto se llamará rag-chile-food-regulations).

### 2. Creación y activación de entorno virtual

Estando dentro del proyecto, ejecute el siguiente comando:

```
python -m venv .venv
```
Esto creará el entorno, ahora debe activarse con:

```
source .venv/bin/activate (Mac)
.venv/Scripts/activate (Windows)
```

### 3. Instalación de dependencias
Se procede a instalar las dependencias necesarias para el proyecto:

```
pip install -r requirements.txt
```

### 4. Carga de chunks y embeddings en base de datos Qdrant

Este proceso considera cuatro fases:
1. Carga de datos.
2. Limpieza de textos y fragmentación (chunking) de textos.
3. Creación de embeddings (vectores) de textos.
4. Almacenamiento en base de datos vectorial, Qdrant.

Para su ejecución:
```
python main_preprocess.py
```

Importante: Esta ejecución se realiza cada vez que se ha iniciado el proyecto desde cero, puesto que la base de datos se almacena en la memoria RAM del equipo desde el cual se ejecuta.

### 5. Despliegue de aplicación con Streamlit

Finalmente, para desplegar y comenzar a usar la aplicación, debe ejecutar el siguiente comando:
```
streamlit run main_app.py
```

## Autores ✒️

_Menciona a todos aquellos que ayudaron a levantar el proyecto desde sus inicios_

* **Paul Contreras** - *Colaborador* 
* **Fernando Echiburu** - *Colaborador*


