import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="EnergiaHogar AI",
                   page_icon="⚡", layout="wide")

st.title("⚡ EnergiaHogar AI")
st.subheader("Inteligencia artificial para tu hogar")
st.write("Calcula, analiza y predice tu consumo eléctrico.")

# ENTRENAMIENTO DEL MODELO
@st.cache_resource
def entrenar():
    np.random.seed(42)
    n = 2000

    consumo = np.random.uniform(10, 800, n)
    personas = np.random.randint(1, 9, n)
    aire = np.random.uniform(0, 15, n)

    # Datos simulados para demostración
    real = (consumo * 1.04 + personas * 2.5
            + aire * 1.2 + np.random.normal(0, 8, n))

    X = pd.DataFrame({
        "consumo": consumo,
        "personas": personas,
        "aire": aire
    })

    x1, x2, y1, y2 = train_test_split(
        X, real, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(x1, y1)
    error = mean_absolute_error(y2, modelo.predict(x2))

    return modelo, error

modelo, error = entrenar()

st.success("Modelo de Machine Learning entrenado.")

# DATOS DEL HOGAR
st.header("🏠 Configuración del hogar")

c1, c2, c3 = st.columns(3)

with c1:
    personas = st.number_input(
        "Personas en casa", 1, 20, 4
    )

with c2:
    tarifa = st.number_input(
        "Precio del kWh (COP)", 1, 10000, 900
    )

with c3:
    dias = st.number_input(
        "Días de facturación", 1, 90, 30
    )

aire = st.number_input(
    "Horas de aire acondicionado al día",
    0.0, 24.0, 0.0, 0.5
)

# ELECTRODOMÉSTICOS
st.header("🔌 Electrodomésticos")

inicial = pd.DataFrame([
    ["Nevera", 150, 24, 1],
    ["Televisor", 100, 5, 1],
    ["Ventilador", 60, 8, 2],
    ["Lavadora", 500, 1, 1],
    ["Bombillos", 10, 6, 5]
], columns=[
    "Electrodoméstico", "Potencia (W)",
    "Horas al día", "Cantidad"
])

tabla = st.data_editor(
    inicial,
    num_rows="dynamic"
)

# VALIDACIÓN Y CÁLCULOS
for col in ["Potencia (W)", "Horas al día", "Cantidad"]:
    tabla[col] = pd.to_numeric(
        tabla[col], errors="coerce"
    ).fillna(0)

tabla["Potencia (W)"] = tabla["Potencia (W)"].clip(lower=0)
tabla["Horas al día"] = tabla["Horas al día"].clip(0, 24)
tabla["Cantidad"] = tabla["Cantidad"].clip(lower=0)

tabla["Consumo (kWh)"] = (
    tabla["Potencia (W)"] *
    tabla["Horas al día"] *
    tabla["Cantidad"] * dias / 1000
)

consumo = tabla["Consumo (kWh)"].sum()

# MACHINE LEARNING
entrada = pd.DataFrame([{
    "consumo": consumo,
    "personas": personas,
    "aire": aire
}])

prediccion = max(0, float(modelo.predict(entrada)[0]))

# RESULTADOS
st.header("📊 Resultados")

a, b, c = st.columns(3)

a.metric("Consumo calculado", f"{consumo:.2f} kWh")
b.metric("Predicción IA", f"{prediccion:.2f} kWh")
c.metric("Factura estimada",
         f"$ {prediccion * tarifa:,.0f} COP")

st.info(
    f"Costo según consumo directo: "
    f"$ {consumo * tarifa:,.0f} COP"
)

st.caption(
    "La predicción es experimental porque el modelo "
    "usa datos simulados. El cálculo directo utiliza "
    "la potencia y horas de uso registradas."
)

# GRÁFICA
st.header("📈 Consumo por electrodoméstico")

if not tabla.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    tabla.set_index("Electrodoméstico")[
        "Consumo (kWh)"
    ].plot(kind="bar", ax=ax)
    ax.set_ylabel("kWh")
    ax.set_xlabel("Aparato")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# RECOMENDACIONES
st.header("💡 Recomendaciones inteligentes")

if not tabla.empty and consumo > 0:
    mayor = tabla.loc[tabla["Consumo (kWh)"].idxmax()]
    st.write(
        f"Tu mayor consumidor es **{mayor['Electrodoméstico']}**, "
        f"con {mayor['Consumo (kWh)']:.2f} kWh."
    )

    st.write("• Revisa las horas de uso de tus aparatos.")
    st.write("• Apaga las luces que no necesites.")
    st.write("• Evita dejar equipos encendidos sin utilizarlos.")
    st.write("• Compara el consumo de tus electrodomésticos.")

# DETALLES DEL MODELO
with st.expander("🤖 Información del Machine Learning"):
    st.write("Algoritmo: Regresión Lineal.")
    st.write("Biblioteca: Scikit-learn.")
    st.write("Registros simulados para entrenamiento: 2.000.")
    st.write(f"Error medio de prueba: {error:.2f} kWh.")
    st.write(
        "El modelo aprende relaciones entre consumo, "
        "personas y uso del aire acondicionado."
    )
    st.warning(
        "Para conseguir predicciones reales se necesitan "
        "historiales de consumo eléctrico medido."
    )

st.caption("EnergiaHogar AI | Proyecto académico de Machine Learning")
