import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv("model/data.csv")


# função para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop("MEDV",axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
    rf_regressor.fit(x, y)
    return rf_regressor

# criando um dataframe
data = get_data()

# treinando o modelo
model = train_model()

# título
st.title('Prediction of the value of real estate')

# subtítulo
st.markdown('Data application to display a machine learning solution for Boston real state prediction problems.')

# verificando o dataset
st.subheader('Selecting only a small set of attributes')

# atributos para serem exibidos por padrão
defaultcols = ["RM","PTRATIO","LSTAT","MEDV"]

# defindo atributos a partir do multiselect
cols = st.multiselect("Attributes", data.columns.tolist(), default=defaultcols)

# exibindo os top 10 registro do dataframe
st.dataframe(data[cols].head(10))


st.subheader('Distribution of properties by price')

# definindo a faixa de valores
faixa_valores = st.slider('Price range', float(data.MEDV.min()), 150., (10.0, 100.0))

# filtrando os dados
dados = data[data['MEDV'].between(left=faixa_valores[0],right=faixa_valores[1])]

# plot a distribuição dos dados
f = px.histogram(dados, x="MEDV", nbins=100, title='Distribution of prices')
f.update_xaxes(title="MEDV")
f.update_yaxes(title="Total properties")
st.plotly_chart(f)


st.sidebar.subheader('Set property attributes for prediction')

# mapeando dados do usuário para cada atributo
crim = st.sidebar.number_input('Crime rate', value=data.CRIM.mean())
indus = st.sidebar.number_input('Proportion of Business Hectares', value=data.CRIM.mean())
chas = st.sidebar.selectbox('Does it border the river?',('Yes','No'))

# transformando o dado de entrada em valor binário
chas = 1 if chas == 'Yes' else 0

nox = st.sidebar.number_input('Concentration of nitric oxide', value=data.NOX.mean())

rm = st.sidebar.number_input('Number of rooms', value=1)

ptratio = st.sidebar.number_input('Student index for teachers',value=data.PTRATIO.mean())

b = st.sidebar.number_input('Proportion of persons with Afro-American descent',value=data.B.mean())

lstat = st.sidebar.number_input('Low status percentage',value=data.LSTAT.mean())

# inserindo um botão na tela
btn_predict = st.sidebar.button('Perform Prediction')

# verifica se o botão foi acionado
if btn_predict:
    result = model.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat]])
    st.subheader('The estimated value of the property is:')
    result = "US $ "+str(round(result[0]*10,2))
    st.write(result)