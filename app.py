
import pickle
import streamlit as st
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')
# App Header
st.header(' Stores Sales Prediction  ')
# Model Import n
model = pickle.load(open('Model\knn.pkl', 'rb'))
standarscalr = pickle.load(open('Model\Sc.pkl', 'rb'))

Item_Weight = st.number_input('Enter a Item_Weight')

Item_Fat_Content = st.selectbox(
    "Enter Item_Fat_Content => Low Fat : 0 | Regular : 1|", (0, 1))

Item_Visibility = st.number_input('Enter a Item_Visibility')

Item_Type = st.selectbox(
    "Enter Item_Type = > Dairy: 4 | Soft Drinks: 14 | Meat: 10 | Fruits and Vegetables: 6 | Household : 9 | Baking Goods : 0 | Snack Foods : 13 | Frozen Foods : 5 | Breakfast : 2 , Health and Hygiene : 8 , Hard Drinks : 7, Canned : 3, Breads : 1, Starchy Foods : 15, Others : 11, Seafood : 12", (4, 14, 10,  6,  9,  0, 13,  5,  2,  8,  7,  3,  1, 15, 11, 12))

Item_MRP = st.number_input('Enter a Item_MRP')

Outlet_Establishment_Year = int(
    st.number_input('Enter a Outlet_Establishment_Year'))

Outlet_Size = st.selectbox(
    "Enter Outlet_Size => Medium: 1 |  No : 3  |  High : 0  | Small : 2 ", (0, 1, 2, 3))

Outlet_Location_Type = st.selectbox(
    "Enter Outlet_Location_Type => 'Tier 1' : 0 | 'Tier 3': 2 | 'Tier 2' : 1 ", (0, 1, 2))
Outlet_Type = st.selectbox(
    "Enter Your Self_Employed Status= > 'Supermarket Type1' : 1 |  'Supermarket Type2' : 2, 'Grocery Store' : 0 ,'Supermarket Type3 : 3'", (0, 1, 2, 3))


x_test = pd.DataFrame([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Establishment_Year,
                      Outlet_Size, Outlet_Location_Type, Outlet_Type]])

x_test_sc = standarscalr.transform(x_test)
result = model.predict(x_test_sc)[0]
if st.button('Predict'):
    st.write(result)
else:
    st.write(".......")

# ['Item_Weight', 'Item_Fat_Content', 'Item_Fat_Content', 'Item_Type',
#  'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size',
#  'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']
