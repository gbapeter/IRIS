import streamlit as st
import pandas as pd
import joblib


model = joblib.load('rfiris.pkl')

st.title('Iris Flower Prediction Application')

st.write('This app takes the features of a flower and predict its name')


form = st.form('Iris flower features')
form.subheader('Please enter the features of the flower')

sepals_length = form.number_input(
		'sepals_length (cm)',
		min_value = 3.0,
		max_value = 8.0,
		value = 5.1
	)

sepals_width = form.number_input(
		'sepals_width (cm)',
		min_value = 1.0,
		max_value = 4.0,
		value = 2.5
	)

petals_length = form.number_input(
		'petals_length (cm)',
		min_value = 0.5,
		max_value = 7.0,
		value = 3.1
	)

petals_width = form.number_input(
		'sepals_width (cm)',
		min_value = 0.2,
		max_value = 3.0,
		value = 1.5
	)

submit_button = form.form_submit_button('Predict')


if submit_button:
	input_data = pd.DataFrame({
		'sepals_length' : [sepals_length],
		'sepals_width' : [sepals_width],
		'petals_length' : [petals_length],
		'petals_width' : [petals_width]
		})
	prediction = model.predict(input_data.values)

	st.subheader('Prediction Result')
	st.success(f'Predicted Species: {prediction[0]}')

