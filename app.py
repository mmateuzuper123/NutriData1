import streamlit as st
import pandas as pd
import re
from openai import OpenAI
from scipy.stats import shapiro

# =====================================
# CONFIGURAÇÃO
# =====================================

API_KEY = st.secrets["OPENAI_API_KEY"]

CSV_FILE = "dados.csv"

# =====================================
# OPENROUTER
# =====================================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# =====================================
# LER CSV
# =====================================

df = pd.read_csv(CSV_FILE, sep=";")

df.columns = df.columns.str.strip()

# =====================================
# CONVERTER NÚMEROS
# =====================================

for coluna in ["Peso", "Altura", "IMC"]:

    if coluna in df.columns:

        df[coluna] = pd.to_numeric(
            df[coluna]
            .astype(str)
            .str.replace(",", "."),
            errors="coerce"
        )

# =====================================
# INTERFACE
# =====================================

st.set_page_config(
    page_title="NutriData+",
    layout="wide"
)

st.title("🩺 NutriData+")

st.markdown("""
Sistema conversacional baseado em IA
para análise clínico-nutricional.
""")

# =====================================
# BASE DE DADOS
# =====================================

with st.expander("Ver Base de Dados"):
    st.write(df.head())

# =====================================
# INPUT
# =====================================

pergunta = st.text_input(
    "Faça uma pergunta clínica:"
)

# =====================================
# PROCESSAMENTO
# =====================================

if pergunta:

    pergunta_original = pergunta
    pergunta = pergunta.lower()

    st.divider()

    st.subheader("Pergunta")
    st.write(pergunta_original)

    # =====================================
    # DETETAR COLUNAS
    # =====================================

    colunas = []

    if "peso" in pergunta:
        colunas.append("Peso")

    if "altura" in pergunta:
        colunas.append("Altura")

    if "imc" in pergunta:
        colunas.append("IMC")

    coluna = colunas[0] if len(colunas) > 0 else None

    # =====================================
    # MÉDIA
    # =====================================

    if (
        "média" in pergunta
        or "media" in pergunta
    ) and coluna:

        media = df[coluna].mean()

        st.subheader("Resultado")

        st.write(f"Média de {coluna}: {media:.2f}")

        st.bar_chart(df[coluna])

    # =====================================
    # MEDIANA
    # =====================================

    elif "mediana" in pergunta and coluna:

        mediana = df[coluna].median()

        st.subheader("Resultado")

        st.write(f"Mediana de {coluna}: {mediana:.2f}")

        st.bar_chart(df[coluna])

    # =====================================
    # MODA
    # =====================================

    elif "moda" in pergunta and coluna:

        moda = df[coluna].mode()[0]

        st.subheader("Resultado")

        st.write(f"Moda de {coluna}: {moda}")

        st.bar_chart(df[coluna])

    # =====================================
    # MÁXIMO
    # =====================================

    elif (
        "máximo" in pergunta
        or "maximo" in pergunta
        or "maior valor" in pergunta
    ) and coluna:

        valor = df[coluna].max()

        st.subheader("Resultado")

        st.write(f"Valor máximo de {coluna}: {valor}")

    # =====================================
    # MÍNIMO
    # =====================================

    elif (
        "mínimo" in pergunta
        or "minimo" in pergunta
        or "menor valor" in pergunta
    ) and coluna:

        valor = df[coluna].min()

        st.subheader("Resultado")

        st.write(f"Valor mínimo de {coluna}: {valor}")

    # =====================================
    # ACIMA DE
    # =====================================

    elif (
        "acima de" in pergunta
        or "maior que" in pergunta
        or "mais de" in pergunta
    ) and coluna:

        numeros = re.findall(r"\d+[.,]?\d*", pergunta)

        if len(numeros) > 0:

            numero = float(
                numeros[0].replace(",", ".")
            )

            resultado = df[df[coluna] > numero]

            st.subheader("Resultado")

            st.write(
                f"{len(resultado)} pessoas "
                f"com {coluna} acima de {numero}"
            )

            st.write(resultado)

            st.bar_chart(resultado[coluna])

    # =====================================
    # ABAIXO DE
    # =====================================

    elif (
        "abaixo de" in pergunta
        or "menor que" in pergunta
        or "menos de" in pergunta
    ) and coluna:

        numeros = re.findall(r"\d+[.,]?\d*", pergunta)

        if len(numeros) > 0:

            numero = float(
                numeros[0].replace(",", ".")
            )

            resultado = df[df[coluna] < numero]

            st.subheader("Resultado")

            st.write(
                f"{len(resultado)} pessoas "
                f"com {coluna} abaixo de {numero}"
            )

            st.write(resultado)

            st.bar_chart(resultado[coluna])

    # =====================================
    # ENTRE
    # =====================================

    elif (
        "entre" in pergunta
        and coluna
    ):

        numeros = re.findall(
            r"\d+[.,]?\d*",
            pergunta
        )

        if len(numeros) >= 2:

            n1 = float(
                numeros[0].replace(",", ".")
            )

            n2 = float(
                numeros[1].replace(",", ".")
            )

            resultado = df[
                (df[coluna] >= n1)
                &
                (df[coluna] <= n2)
            ]

            st.subheader("Resultado")

            st.write(
                f"{len(resultado)} pessoas "
                f"com {coluna} entre "
                f"{n1} e {n2}"
            )

            st.write(resultado)

            st.bar_chart(resultado[coluna])

    # =====================================
    # CORRELAÇÃO
    # =====================================

    elif (
        "correlação" in pergunta
        or "correlacao" in pergunta
        or "relação" in pergunta
        or "relacao" in pergunta
        or "associação" in pergunta
        or "associacao" in pergunta
    ):

        if len(colunas) >= 2:

            dados = df[colunas].dropna()

            correlacao = dados[
                colunas[0]
            ].corr(
                dados[colunas[1]]
            )

            st.subheader("Resultado")

            st.write(
                f"A correlação entre "
                f"{colunas[0]} e "
                f"{colunas[1]} é "
                f"{correlacao:.2f}"
            )

            if correlacao > 0.7:
                interpretacao = (
                    "Correlação forte positiva"
                )

            elif correlacao > 0.3:
                interpretacao = (
                    "Correlação moderada positiva"
                )

            elif correlacao > 0:
                interpretacao = (
                    "Correlação fraca positiva"
                )

            elif correlacao < -0.7:
                interpretacao = (
                    "Correlação forte negativa"
                )

            else:
                interpretacao = (
                    "Correlação fraca"
                )

            st.write(
                f"Interpretação: "
                f"{interpretacao}"
            )

            st.scatter_chart(
                dados,
                x=colunas[0],
                y=colunas[1]
            )

        else:

            st.error(
                "Indique duas variáveis."
            )

    # =====================================
    # SHAPIRO-WILK
    # =====================================

    elif (
        "normalidade" in pergunta
        or "shapiro" in pergunta
    ):

        st.subheader(
            "Teste de Normalidade"
        )

        resultados = []

        for var in [
            "Peso",
            "Altura",
            "IMC"
        ]:

            dados = df[var].dropna()

            stat, p = shapiro(dados)

            resultados.append({
                "Variável": var,
                "Estatística": round(stat, 4),
                "Valor-p": round(p, 4),
                "Normal?":
                    "Sim"
                    if p > 0.05
                    else "Não"
            })

        resultados_df = pd.DataFrame(
            resultados
        )

        st.write(resultados_df)

    # =====================================
    # DISPERSÃO
    # =====================================

    elif (
        "dispersão" in pergunta
        or "dispersao" in pergunta
        or "variabilidade" in pergunta
    ):

        desvios = {
            "Peso": df["Peso"].std(),
            "Altura": df["Altura"].std(),
            "IMC": df["IMC"].std()
        }

        maior = max(
            desvios,
            key=desvios.get
        )

        st.subheader("Resultado")

        st.write(
            f"A variável com maior "
            f"dispersão é {maior}"
        )

        st.write(desvios)

    # =====================================
    # HOMOGENEIDADE
    # =====================================

    elif (
        "homogénea" in pergunta
        or "homogenea" in pergunta
        or "homogéneo" in pergunta
        or "homogeneo" in pergunta
    ):

        desvios = {
            "Peso": df["Peso"].std(),
            "Altura": df["Altura"].std(),
            "IMC": df["IMC"].std()
        }

        menor = min(
            desvios,
            key=desvios.get
        )

        st.subheader("Resultado")

        st.write(
            f"A variável mais "
            f"homogénea é {menor}"
        )

        st.write(desvios)

    # =====================================
    # OUTLIERS
    # =====================================

    elif "outlier" in pergunta:

        if coluna:

            q1 = df[coluna].quantile(0.25)
            q3 = df[coluna].quantile(0.75)

            iqr = q3 - q1

            limite_inf = q1 - 1.5 * iqr
            limite_sup = q3 + 1.5 * iqr

            outliers = df[
                (df[coluna] < limite_inf)
                |
                (df[coluna] > limite_sup)
            ]

            st.subheader("Resultado")

            st.write(
                f"Foram encontrados "
                f"{len(outliers)} outliers "
                f"em {coluna}"
            )

            st.write(outliers)

        else:

            st.error(
                "Indique uma variável."
            )

    # =====================================
    # RESUMO ESTATÍSTICO
    # =====================================

    elif (
        "análise" in pergunta
        or "analise" in pergunta
        or "resumo" in pergunta
        or "achados" in pergunta
        or "conclusões" in pergunta
        or "conclusoes" in pergunta
    ):

        st.subheader(
            "Resumo Estatístico"
        )

        st.write(df.describe())

    # =====================================
    # CONTAGEM TOTAL
    # =====================================

    elif (
        "quantas pessoas" in pergunta
        or "número de pessoas" in pergunta
    ):

        st.subheader("Resultado")

        st.write(
            f"Número total de pessoas: "
            f"{len(df)}"
        )

    # =====================================
    # GPT
    # =====================================

    else:

        try:

            resposta = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",
    messages=[
                    {
                        "role": "system",
                        "content": """
                        És especialista em
                        nutrição clínica hospitalar.

                        Responde de forma
                        científica, resumida
                        e profissional.
                        """
                    },
                    {
                        "role": "user",
                        "content": pergunta_original
                    }
                ]
            )

            texto = resposta.choices[
                0
            ].message.content

            st.subheader("Resposta IA")

            st.write(texto)

        except Exception as e:

            st.error(
                f"Erro OpenRouter: {e}"
            )

# =====================================
# RODAPÉ
# =====================================

st.divider()

st.caption(
    "Protótipo académico "
    "NutriData+ baseado em IA clínica."
)
