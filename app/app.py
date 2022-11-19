import streamlit as st
from src.pred_gen import PredictionGenerator


st.set_page_config(page_title='Airbnb Price Predictor')
st.markdown('<h1 style="text-align:center; color:#ff5a5f">Airbnb Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center; color:#ff5a5f">You are a NYC Airbnb host? See how much you can charge!</h3>', unsafe_allow_html=True)

with st.form('input'):

    lat_col, lon_col = st.columns(2)
    with lat_col:
        latitude = st.number_input('Enter the latitude of your apartment:', value=40.69376916322678, step=.00000000000001, format="%.12f")
    with lon_col:
        longitude = st.number_input('Enter the longitude of your apartment:', value=-73.99399346350506, step=.00000000000001, format="%.12f")

    neigh_col, room_type_col = st.columns(2)
    with neigh_col:
        neighborhood = st.selectbox('Choose the neighborhood:', ['Brooklyn', 'Manhattan', 'Other'])
    with room_type_col:
        room_type_col = st.selectbox('Do you rent out a room or a whole place?', ['Private Room', 'Entire Home'])

    min_nights_col, avail_col = st.columns(2)
    with min_nights_col:
        min_nights = st.number_input('What is the minimum nights required for a booking?', 1, step=1)
    with avail_col:
        availability = st.number_input('How many days per year is your place available?', 0, 365, step=1, value=365)
    
    exp_col, reviews_col, count_col = st.columns(3)
    with exp_col:
        experience = st.number_input('How long (in months) have you been a host?', 1, step=1)
    with reviews_col:
        reviews = st.number_input('How many reviews does your apartment have?', step=1)
    with count_col:
        listings = st.number_input('How many places do you list on Airbnb?', step=1, min_value=1)

    input_submit = st.form_submit_button('Get a price estimate!')

    if input_submit:
        pg = PredictionGenerator(latitude, longitude, neighborhood, room_type_col, min_nights, availability, experience, reviews, listings)
        pg.prepare_input()
        prediction = pg.get_prediction()
        st.metric('Estimated price per night!', round(prediction[0], 2))