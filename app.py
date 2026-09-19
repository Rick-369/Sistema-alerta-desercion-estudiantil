"""
Sistema Inteligente de Alerta Temprana para la Predicción de la Deserción Estudiantil

"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Configuración de la página

st.set_page_config(
    page_title="Alerta Temprana de Deserción Estudiantil",
    page_icon="🎓",
    layout="centered",
)

# Carga de artefactos del modelo

@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load("modelo_desercion.pkl")
    feature_names = joblib.load("feature_names.pkl")
    medianas = joblib.load("medianas_variables.pkl")
    return modelo, feature_names, medianas


modelo, feature_names, medianas = cargar_artefactos()


# Codificación del Diccionario

# Variable "Estado Civil" se codifica

MARITAL_STATUS = {
    1: "Soltero/a", 2: "Casado/a", 3: "Viudo/a",
    4: "Divorciado/a", 5: "Unión de hecho", 6: "Separado/a legalmente",
}

# Definicion de rangos de las siguientes variables:
COURSE = {i: f"Código {i}" for i in range(1, 18)}                # 1-17
APPLICATION_MODE = {i: f"Código {i}" for i in range(1, 19)}       # 1-18
PREVIOUS_QUALIFICATION = {i: f"Código {i}" for i in range(1, 18)}  # 1-17
NACIONALITY = {i: f"Código {i}" for i in range(1, 22)}            # 1-21
MOTHER_QUALIFICATION = {i: f"Código {i}" for i in range(1, 30)}    # 1-29
FATHER_QUALIFICATION = {i: f"Código {i}" for i in range(1, 35)}    # 1-34
MOTHER_OCCUPATION = {i: f"Código {i}" for i in range(1, 33)}       # 1-32
FATHER_OCCUPATION = {i: f"Código {i}" for i in range(1, 47)}       # 1-46

# Codificación de variables de SI o No

SI_NO = {1: "Sí", 0: "No"}

#Diccionario de la correlación con la deserción

CORRELACION_DESERCION = {
    "Curricular units 1st sem (grade)": -0.481,
    "Curricular units 1st sem (approved)": -0.479,
    "Tuition fees up to date": -0.429,
    "Age at enrollment": 0.254,
    "Scholarship holder": -0.245,
    "Debtor": 0.229,
    "Gender": 0.204,
    "Application mode": 0.189,
    "Curricular units 1st sem (enrolled)": -0.125,
    "Curricular units 1st sem (evaluations)": -0.090,
    "Father's occupation": -0.080,
    "Mother's occupation": -0.069,
    "GDP": -0.046,
    "Unemployment rate": 0.013,
    "Course": 0.000,
}

#Traducción de los nombres de las variables al español
NOMBRES_ES = {
    "Curricular units 1st sem (approved)": "Unidades aprobadas (1er sem.)",
    "Curricular units 1st sem (grade)": "Nota promedio (1er sem.)",
    "Tuition fees up to date": "Matrícula al día",
    "Age at enrollment": "Edad de matrícula",
    "Curricular units 1st sem (evaluations)": "Unidades evaluadas (1er sem.)",
    "Debtor": "Tiene deudas",
    "Scholarship holder": "Tiene beca",
    "Course": "Curso / carrera",
    "Curricular units 1st sem (enrolled)": "Unidades matriculadas (1er sem.)",
    "Application mode": "Modalidad de aplicación",
    "Gender": "Género",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Unemployment rate": "Tasa de desempleo",
    "GDP": "PIB",
}

# Con Selectbox 

def selectbox_codigo(label, opciones_dict, valor_defecto, key, help_text=None):

    codigos = list(opciones_dict.keys())
    if valor_defecto not in codigos:
        codigos = [valor_defecto] + codigos
    idx_defecto = codigos.index(valor_defecto)
    return st.selectbox(
        label,
        options=codigos,
        index=idx_defecto,
        format_func=lambda c: opciones_dict.get(c, f"Código {c}"),
        key=key,
        help=help_text,
    )


def mediana(nombre):
    return medianas.get(nombre, 0)


# Se define a la Moda como valor más frecuente para las variables que queda OCULTA en el formulario. 

MODA_VARIABLES = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 1,
    "Father's qualification": 27,
    "Mother's occupation": 10,
    "Father's occupation": 10,
    "Displaced": 1,
    "Educational special needs": 0,
    "Gender": 0,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (without evaluations)": 0,
    "Unemployment rate": 7.6,
    "Inflation rate": 1.4,
    "GDP": 0.32,
}


def moda(nombre):
    return MODA_VARIABLES.get(nombre, 0)



# Encabezado

st.title("🎓 Sistema Inteligente de Alerta Temprana para la Predicción de la Deserción Estudiantil")


st.divider()

# ---------------------------------------------------------------------------
# Formulario de entrada
# ---------------------------------------------------------------------------
with st.form("formulario_estudiante"):

    # -------------------------------------------------------------------
    # Las siguientes 9 variables son, en ese orden, las de mayor
    # importancia para el modelo (modelo.feature_importances_) — hay un
    # corte natural en el ranking justo después de la novena (la décima,
    # "Application mode", ya pesa menos de la mitad que la novena). Por
    # eso son las únicas visibles en el formulario principal; el resto
    # queda oculto y se autocompleta con la moda.
    # -------------------------------------------------------------------
    st.subheader("🎯 Datos académicos y de admisión")
    c1, c2 = st.columns(2)
    with c1:
        course = selectbox_codigo("Curso / carrera", COURSE, int(mediana("Course")), "course")
    with c2:
        age = st.number_input(
            "Edad de matrícula", min_value=16, max_value=80,
            value=int(mediana("Age at enrollment")), step=1, key="age"
        )

    st.subheader("📊 Rendimiento — 1er semestre")
    c3, c4 = st.columns(2)
    with c3:
        cu1_enrolled = st.number_input(
            "Unidades curriculares matriculadas", min_value=0, max_value=30,
            value=int(mediana("Curricular units 1st sem (enrolled)")), step=1, key="cu1_enrolled"
        )
        cu1_approved = st.number_input(
            "Unidades curriculares aprobadas", min_value=0, max_value=30,
            value=int(mediana("Curricular units 1st sem (approved)")), step=1, key="cu1_approved"
        )
    with c4:
        cu1_evaluations = st.number_input(
            "Unidades curriculares evaluadas", min_value=0, max_value=45,
            value=int(mediana("Curricular units 1st sem (evaluations)")), step=1, key="cu1_evaluations"
        )
        cu1_grade = st.number_input(
            "Nota promedio (0-20)", min_value=0.0, max_value=20.0,
            value=float(mediana("Curricular units 1st sem (grade)")), step=0.1, key="cu1_grade"
        )

    st.subheader("💰 Situación económica")
    c5, c6, c7 = st.columns(3)
    with c5:
        tuition_up_to_date = selectbox_codigo(
            "¿Matrícula al día?", SI_NO, int(mediana("Tuition fees up to date")), "tuition"
        )
    with c6:
        debtor = selectbox_codigo("¿Tiene deudas?", SI_NO, int(mediana("Debtor")), "debtor")
    with c7:
        scholarship = selectbox_codigo(
            "¿Tiene beca?", SI_NO, int(mediana("Scholarship holder")), "scholarship"
        )

    # -------------------------------------------------------------------
    # Variables adicionales (colapsadas): las 19 restantes, todas de
    # menor peso para el modelo. Se autocompletan con la MODA (valor más
    # frecuente en dataset_original.csv de cada variable) en vez de la
    # mediana, y quien quiera más precisión las puede ajustar aquí.
    # -------------------------------------------------------------------
    with st.expander("➕ Variables adicionales (menor impacto en el modelo)"):
        cc1, cc2 = st.columns(2)
        with cc1:
            marital_status = selectbox_codigo(
                "Estado civil", MARITAL_STATUS, int(moda("Marital status")), "marital_status"
            )
            application_mode = selectbox_codigo(
                "Modalidad de aplicación", APPLICATION_MODE,
                int(moda("Application mode")), "application_mode"
            )
            nacionality = selectbox_codigo(
                "Nacionalidad", NACIONALITY, int(moda("Nacionality")), "nacionality"
            )
            displaced = selectbox_codigo(
                "¿Estudiante desplazado de su domicilio?", SI_NO,
                int(moda("Displaced")), "displaced"
            )
            special_needs = selectbox_codigo(
                "¿Necesidades educativas especiales?", SI_NO,
                int(moda("Educational special needs")), "special_needs"
            )
            international = selectbox_codigo(
                "¿Estudiante internacional?", SI_NO, int(moda("International")), "international"
            )
            application_order = st.number_input(
                "Orden de aplicación (0=primera opción, 9=última)",
                min_value=0, max_value=9, value=int(moda("Application order")),
                step=1, key="application_order"
            )
            attendance = selectbox_codigo(
                "Horario", {1: "Diurno", 0: "Nocturno"},
                int(moda("Daytime/evening attendance")), "attendance"
            )
        with cc2:
            gender = selectbox_codigo(
                "Género", {1: "Masculino", 0: "Femenino"}, int(moda("Gender")), "gender"
            )
            previous_qualification = selectbox_codigo(
                "Calificación previa", PREVIOUS_QUALIFICATION,
                int(moda("Previous qualification")), "previous_qualification"
            )
            mother_qualification = selectbox_codigo(
                "Calificación de la madre", MOTHER_QUALIFICATION,
                int(moda("Mother's qualification")), "mother_qualification"
            )
            father_qualification = selectbox_codigo(
                "Calificación del padre", FATHER_QUALIFICATION,
                int(moda("Father's qualification")), "father_qualification"
            )
            mother_occupation = selectbox_codigo(
                "Ocupación de la madre", MOTHER_OCCUPATION,
                int(moda("Mother's occupation")), "mother_occupation"
            )
            father_occupation = selectbox_codigo(
                "Ocupación del padre", FATHER_OCCUPATION,
                int(moda("Father's occupation")), "father_occupation"
            )
            cu1_credited = st.number_input(
                "Unidades curriculares acreditadas", min_value=0, max_value=30,
                value=int(moda("Curricular units 1st sem (credited)")), step=1, key="cu1_credited"
            )
            cu1_no_eval = st.number_input(
                "Unidades curriculares sin evaluación", min_value=0, max_value=30,
                value=int(moda("Curricular units 1st sem (without evaluations)")),
                step=1, key="cu1_no_eval"
            )
            unemployment = st.number_input(
                "Tasa de desempleo (%)", min_value=0.0, max_value=30.0,
                value=float(moda("Unemployment rate")), step=0.1, key="unemployment"
            )
            inflation = st.number_input(
                "Tasa de inflación (%)", min_value=-10.0, max_value=20.0,
                value=float(moda("Inflation rate")), step=0.1, key="inflation"
            )
            gdp = st.number_input(
                "PIB (variación)", min_value=-10.0, max_value=10.0,
                value=float(moda("GDP")), step=0.01, key="gdp"
            )

    submitted = st.form_submit_button("🔍 Predecir riesgo de deserción", use_container_width=True)

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
if submitted:
    entrada = {
        "Marital status": marital_status,
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Daytime/evening attendance": attendance,
        "Previous qualification": previous_qualification,
        "Nacionality": nacionality,
        "Mother's qualification": mother_qualification,
        "Father's qualification": father_qualification,
        "Mother's occupation": mother_occupation,
        "Father's occupation": father_occupation,
        "Displaced": displaced,
        "Educational special needs": special_needs,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_up_to_date,
        "Gender": gender,
        "Scholarship holder": scholarship,
        "Age at enrollment": age,
        "International": international,
        "Curricular units 1st sem (credited)": cu1_credited,
        "Curricular units 1st sem (enrolled)": cu1_enrolled,
        "Curricular units 1st sem (evaluations)": cu1_evaluations,
        "Curricular units 1st sem (approved)": cu1_approved,
        "Curricular units 1st sem (grade)": cu1_grade,
        "Curricular units 1st sem (without evaluations)": cu1_no_eval,
        "Unemployment rate": unemployment,
        "Inflation rate": inflation,
        "GDP": gdp,
    }

    # Se ordena exactamente como feature_names para respetar el orden con el que se entrenó
    df_entrada = pd.DataFrame([entrada])[feature_names]

    pred = modelo.predict(df_entrada)[0]
    proba = modelo.predict_proba(df_entrada)[0][1]

    st.divider()
    st.subheader("Resultado")

    col_resultado, col_gauge = st.columns([1, 1])

    with col_resultado:
        if pred == 1:
            st.error(f"⚠️ **Riesgo de deserción** — probabilidad estimada: **{proba:.1%}**")
            st.write(
                "El modelo identifica a este estudiante dentro del grupo de riesgo. "
                "Se recomienda activar seguimiento tutorial y verificar apoyo académico/económico."
            )
        else:
            st.success(f"✅ **Bajo riesgo de deserción** — probabilidad estimada: **{proba:.1%}**")
            st.write("El modelo no identifica señales tempranas de riesgo relevantes para este estudiante.")

    with col_gauge:
        st.metric("Probabilidad de deserción", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

    st.divider()
    st.subheader("🔎 Variables más influyentes del modelo (top 15)")
    importancias = pd.Series(modelo.feature_importances_, index=feature_names)
    top15 = importancias.sort_values(ascending=False).head(15)
    top15_es = top15.rename(index=lambda n: NOMBRES_ES.get(n, n))
    st.bar_chart(top15_es)
    st.caption(
        "Importancia relativa de cada variable dentro del Random Forest "
        "(cuánto reduce la impureza de los árboles al dividir por esa variable)."
    )

    with st.expander("Ver todos los valores enviados al modelo"):
        st.dataframe(df_entrada.T.rename(columns={0: "Valor"}))

st.divider()
