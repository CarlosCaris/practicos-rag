from typing import Any


def launch_header(st: Any) -> None:
    st.title("Actividad 01")
    st.subheader("Carga de Archivos")
    st.write(
        "En esta sección se cargará un archivo pdf, se preprocesará y se guardará en un Bucket. Detallaremos todo el flujo paso a paso."
    )