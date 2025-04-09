import json

import requests

import streamlit as st


def _get_prediction(inputs_data):
    """
    Função para obter a previsão do modelo.
    """
    url = "http://localhost:5001/invocations"
    data = {
        "inputs": [list(inputs_data.values())]
    }
    response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        return response.json()["predictions"][0]
    else:
        st.error("Erro ao obter a previsão.")
        return None

st.set_page_config(page_title='Kobe Bryant Shot Predictor')
st.title('Kobe Bryant Shot Predictor')

prediction = None

with st.form("kobe_shot_predictor_form"):
    st.header("Parâmetros do Arremesso")
    st.write("Insira os dados do arremesso para obter uma previsão")

    left_column, right_column = st.columns(2)

    with left_column:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=33.8063, key='latitude')
        minutes_remaining = st.number_input("Minutes Remaining", min_value=0.0, max_value=50.0, value=10.0, key='minutes_remaining')
        shot_distance = st.number_input("Shot Distance", min_value=0.0, max_value=1000.0, value=10.0, key='shot_distance')

    with right_column:
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.3638, key='longitude')
        period = st.number_input("Period", min_value=0.0, max_value=50.0, value=10.0, key='period')
        playoffs = st.selectbox("Playoffs", options=["Sim", "Não"], index=0)


    submitted = st.form_submit_button("Prever Resultado", use_container_width=True)
    inputs_data = {
        "lat": latitude,
        "lng": longitude,
        "minutes_remaining": minutes_remaining,
        "period": period,
        "playoffs": 1 if playoffs == "Sim" else 0,
        "shot_distance": shot_distance
    }

    if submitted:
        prediction = _get_prediction(inputs_data)

if prediction is not None:
    with st.container(border=True):
        left_column, center_column, right_column  = st.columns([2, 6, 2])
        icon = "✅" if prediction == 1 else "❌"
        result_title = "Acerto" if prediction == 1 else "Errou"
        result_description = "Kobe teria acertado este arremesso!" if prediction == 1 else "Kobe teria errado este arremesso!"

        with center_column:
            st.markdown(
                f"""
                <div style="
                    width: 50px;
                    height: 50px;
                    margin: 16px auto 16px auto;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 48px;
                ">
                    {icon}
                </div>
                <div style="text-align: center;">
                    <h3>Resultado: {result_title}</h3>
                    <p>{result_description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

