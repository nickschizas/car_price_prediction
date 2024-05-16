import streamlit as st
import pandas as pd
import numpy as np
from pickle import load

st.set_page_config(layout="centered", page_title='Car Price Prediction', page_icon=':rocket:')

about_text = """
This is a ML web app designed for predicting car prices.
You can choose your desired features to receive a prediction.
Both the training and testing data were web-scraped from web.
Navigate to the second page for an EDA on these data.
Our model development and data are frequently updated to ensure accuracy.
"""

contact_text = """
Feel free to fork this app or contact:<br>
n.schizas@outlook.com
"""
st.sidebar.markdown(f'<em>{about_text}<em>', unsafe_allow_html=True)
st.sidebar.caption(f'<em>{contact_text}<em>', unsafe_allow_html=True)

# @st.cache_data
# def load_model(file_name):
#     models_path = r'./models/'
#     file=open(models_path+file_name, 'rb')
#     return load(file)

@st.cache_data
def transform_input(input):
    input_new = input.copy()
    for key in input.keys():
        if key in ['Name', 'GasType', 'GearBox']:
            input_new.update({f'{key}_{input[key]}' : True})
            input_new.pop(key)

    input_columns = ['Klm', 'CubicCapacity', 'Horsepower', 'Age', 'Name_Alfa-Romeo','Name_Audi', 'Name_Bmw', 'Name_Chevrolet', 'Name_Citroen', 'Name_DS',
       'Name_Dacia', 'Name_Daewoo', 'Name_Daihatsu', 'Name_Fiat', 'Name_Ford','Name_Honda', 'Name_Hyundai', 'Name_Isuzu', 'Name_Jaguar', 'Name_Jeep',
       'Name_Kia', 'Name_Lancia', 'Name_Land-Rover', 'Name_Lexus','Name_Mazda', 'Name_Mercedes-Benz', 'Name_Mini-Cooper','Name_Mitsubishi', 'Name_Nissan', 'Name_Opel', 'Name_Peugeot',
       'Name_Porsche', 'Name_Renault', 'Name_Saab', 'Name_Seat', 'Name_Skoda', 'Name_Smart', 'Name_Subaru', 'Name_Suzuki', 'Name_Toyota', 'Name_Volkswagen', 'Name_Volvo', 'GasType_Αέριο(lpg) - βενζίνη',
       'GasType_Βενζίνη', 'GasType_Πετρέλαιο','GasType_Υβριδικό plug-in βενζίνη', 'GasType_Υβριδικό plug-in πετρέλαιο', 'GasType_Υβριδικό βενζίνη',
       'GasType_Υβριδικό πετρέλαιο', 'GasType_Φυσικό αέριο(cng) - βενζίνη', 'GearBox_Manual', 'GearBox_Αυτόματο', 'GearBox_Ημιαυτόματο']
    input_df = pd.DataFrame(columns=input_columns)
    input_data_df = pd.DataFrame(input_new, index=[0])
    
    return pd.concat([input_df,input_data_df]).fillna(False)

@st.cache_data
def make_model():
    # Load & preprocess the data
    DATA_PATH = './data/data_clean_20240509.csv'
    data = pd.read_csv(DATA_PATH, sep=';')
    data_processed = pd.get_dummies(data, prefix_sep = '_')
    X = data_processed.drop('Price', axis=1)
    y = data_processed['Price']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)
    # Import packages
    from sklearn.ensemble import RandomForestRegressor
    # Define and fit the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def prediction(model, input):
    # Transform input data
    input_data = transform_input(input)
    prediction = model.predict(input_data)[0]
    return prediction

# def prediction(input):
#     # Import packages
#     from sklearn.ensemble import RandomForestRegressor
#     from sklearn.model_selection import train_test_split
#     # Transform input data
#     input_data = transform_input(input)
#     # Load & preprocess the data
#     DATA_PATH = './data/data_clean_20240509.csv'
#     data = pd.read_csv(DATA_PATH, sep=';')
#     data_processed = pd.get_dummies(data, prefix_sep = '_')
#     X = data_processed.drop('Price', axis=1)
#     y = data_processed['Price']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)
#     # Define and fit the model
#     rfr = RandomForestRegressor()
#     rfr.fit(X_train, y_train)
#     prediction = rfr.predict(input_data)[0]
#     return prediction

# model = load_model('RandomForestRegressor.pkl')

brands = ['Citroen', 'Mercedes-Benz', 'Toyota', 'Mitsubishi', 'Jaguar',
       'Hyundai', 'Land-Rover', 'Bmw', 'Kia', 'Skoda', 'Opel',
       'Volkswagen', 'Daihatsu', 'Nissan', 'Renault', 'Smart', 'Seat',
       'Peugeot', 'Mazda', 'Fiat', 'Audi', 'Ford', 'Lexus', 'Volvo',
       'Jeep', 'Mini-Cooper', 'Maserati', 'Ferrari', 'Suzuki',
       'Chevrolet', 'Dacia', 'Lancia', 'Porsche', 'Alfa', 'Aston', 'DS',
       'Daewoo', 'Subaru', 'Isuzu', 'Caterham', 'Honda', 'Wartburg',
       'Autobianchi', 'Lamborghini', 'McLaren', 'TVR', 'Piaggio',
       'Infiniti', 'Saab', 'Mg', 'Club', 'Triumph', 'Cadillac', 'Abarth',
       'Morgan', 'Chrysler', 'Maybach', 'Lada', 'SsangYong', 'Dodge',
       'Buick', 'Rolls', 'Bentley', 'Hummer', 'Austin', 'Corvette',
       'Plymouth', 'DR', 'Lotus', 'Rover', 'Lynk&Co', 'Pontiac', 'Iveco',
       'Cupra', 'Lincoln']

gear_box = ['Manual', 'Αυτόματο', 'Ημιαυτόματο']

gas_types = ['Πετρέλαιο', 'Βενζίνη', 'Αέριο(lpg) - βενζίνη', 'Υβριδικό βενζίνη',
       'Υβριδικό plug-in βενζίνη', 'Φυσικό αέριο(cng) - βενζίνη',
       'Υβριδικό πετρέλαιο', 'Υβριδικό plug-in πετρέλαιο']

with st.columns([1,2.6,1])[1]:
    st.title('Car Price Prediction', anchor=False)

with st.form(key='input'):
    form_col = st.columns(2)
    with form_col[0]:
       name = st.selectbox('Car Brand', options=brands)
       gearbox = st.selectbox('Gear Box Type', options=gear_box)
       age = st.number_input('Age', min_value=0, value=8)
       cc = st.number_input('CubicCapacity', min_value=0, value=1300)

    with form_col[1]:
       gastype = st.selectbox('GasType', options=gas_types)
       klm = st.number_input('Kilometers', min_value=0, value=80000)
       hp = st.number_input('Horsepower', min_value=0, value=100)

    
    predict = st.form_submit_button('Predict')
    if predict:
       input = {'Klm':klm, 'CubicCapacity':cc, 'Horsepower':hp, 'Age':age, 'Name':name, 'GasType':gastype, 'GearBox':gearbox}
       st.session_state['pred'] = prediction(make_model(), input)

if 'pred' in st.session_state:
       pred = st.session_state['pred'].copy()
       st.subheader(f'Prediction : {"{:,.0f}".format(pred)} €', anchor=False)
