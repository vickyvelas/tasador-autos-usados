FROM astrocrpublic.azurecr.io/runtime:3.3-7

# Para que el DAG pueda hacer `from autos import ...` (código en include/autos)
ENV PYTHONPATH="/usr/local/airflow/include:${PYTHONPATH}"
