"""El DAG carga sin errores de importación y tiene la forma esperada."""
from airflow.models import DagBag

DAG_ID = "tp1_5K09_03_autos_mensual"


def _dag():
    bag = DagBag(dag_folder="dags", include_examples=False)
    assert not bag.import_errors, bag.import_errors
    return bag.get_dag(DAG_ID)


def test_tareas_y_dependencias():
    dag = _dag()
    assert {t.task_id for t in dag.tasks} == {
        "crear_carpetas_trabajo", "validar_disponibilidad_fuentes", "obtener_marcas_actuales",
        "elegir_ruta_datos", "chequear_refresh_mensual", "usar_respaldo_congelado",
        "cosecha_bronce_carone", "cosecha_bronce_deruedas", "consolidar_capa_plata",
        "validar_calidad_dataset", "generar_entregable_zip",
    }
    validar = dag.get_task("validar_calidad_dataset")
    # las dos ramas llegan a la validación
    assert validar.upstream_task_ids == {"consolidar_capa_plata", "usar_respaldo_congelado"}
    assert dag.default_args["retries"] >= 2
