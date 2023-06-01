import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
df = pd.read_csv('CountryLE_EXP_UHC.csv')

# Take the logarithm of the x-axis values
df['log_x'] = np.log(df['2019 Health Expenditure Per Capita'])
df['Universal Health Care Coverage'] = df['UHC'].apply(lambda x: 'Yes' if x == 1 else 'No')

# Create a linear regression model
model = LinearRegression()
# Fit the model using the transformed data
model.fit(df[['log_x']], df['2019 Life Expectancy at Birth'])

# Create the first plot
fig1 = px.scatter(df, 
                  x='2019 Health Expenditure Per Capita', 
                  y='2019 Life Expectancy at Birth', 
                  hover_name='Location',
                  color='Universal Health Care Coverage')
# Add the linear regression line to the first plot
x_range = np.linspace(df['log_x'].min(), df['log_x'].max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))
fig1.add_trace(
    go.Scatter(
        x=np.exp(x_range),
        y=y_range,
        mode='lines',
        name='Line of Best Fit',
        line=dict(color='grey', width=2)
    )
)
# Customize the first plot's layout
fig1.update_layout(
    xaxis=dict(showgrid=True, showticklabels=True),
    yaxis=dict(showgrid=True, showticklabels=True),
    xaxis_title='2019 Health Expenditure Per Capita (international dollars)',
    yaxis_title='2019 Life Expectancy at Birth (years)',
    title="Health Expenditure and Life Expectancy of the World's Countries"
)

# Create the second plot with log x-axis
fig2 = px.scatter(df,
                  x='log_x',
                  y='2019 Life Expectancy at Birth',
                  hover_name='Location',
                  color='Universal Health Care Coverage')
# Add the linear regression line to the second plot
x_range_log = np.linspace(df['log_x'].min(), df['log_x'].max(), 100)
y_range_log = model.predict(x_range_log.reshape(-1, 1))
fig2.add_trace(
    go.Scatter(
        x=x_range_log,
        y=y_range_log,
        mode='lines',
        name='Line of Best Fit',
        line=dict(color='grey', width=2)
    )
)
# Customize the second plot's layout
fig2.update_layout(
    xaxis=dict(showgrid=True, showticklabels=True),
    yaxis=dict(showgrid=True, showticklabels=True),
    xaxis_title='Log of 2019 Health Expenditure Per Capita (international dollars)',
    yaxis_title='2019 Life Expectancy at Birth (years)',
    title="Health Expenditure (log scale) and Life Expectancy of the World's Countries"
)
# Calculate model evaluation metrics
y_pred = model.predict(df[['log_x']])
r2 = r2_score(df['2019 Life Expectancy at Birth'], y_pred)
mse = mean_squared_error(df['2019 Life Expectancy at Birth'], y_pred)

# Calculate mean and median for universal health coverage vs. no universal health coverage
mean_with_uhc = df.loc[df['Universal Health Care Coverage'] == 'Yes', ['2019 Health Expenditure Per Capita', '2019 Life Expectancy at Birth']].mean()
median_with_uhc = df.loc[df['Universal Health Care Coverage'] == 'Yes', ['2019 Health Expenditure Per Capita', '2019 Life Expectancy at Birth']].median()
mean_without_uhc = df.loc[df['Universal Health Care Coverage'] == 'No', ['2019 Health Expenditure Per Capita', '2019 Life Expectancy at Birth']].mean()
median_without_uhc = df.loc[df['Universal Health Care Coverage'] == 'No', ['2019 Health Expenditure Per Capita', '2019 Life Expectancy at Birth']].median()

# Create Streamlit buttons to enable each plot
if st.button('Health expenditure vs. Life Expectancy'):
    st.plotly_chart(fig1)
    

if st.button('Health expenditure (log scale) vs. Life Expectancy'):
    st.plotly_chart(fig2)
    
# Display statistics for universal healthcare coverage in a table
st.markdown("<h2 style='text-align: center;'>Statistics for Universal Health Care Coverage:</h2>", unsafe_allow_html=True)
data = {
    'Metrics': ['Mean Health Expenditure', 'Median Health Expenditure', 'Mean Life Expectancy', 'Median Life Expectancy'],
    'With Universal Health Care Coverage': [
        f"{mean_with_uhc['2019 Health Expenditure Per Capita']:.2f}",
        f"{median_with_uhc['2019 Health Expenditure Per Capita']:.2f}",
        f"{mean_with_uhc['2019 Life Expectancy at Birth']:.2f}",
        f"{median_with_uhc['2019 Life Expectancy at Birth']:.2f}"
    ],
    'Without Universal Health Care Coverage': [
        f"{mean_without_uhc['2019 Health Expenditure Per Capita']:.2f}",
        f"{median_without_uhc['2019 Health Expenditure Per Capita']:.2f}",
        f"{mean_without_uhc['2019 Life Expectancy at Birth']:.2f}",
        f"{median_without_uhc['2019 Life Expectancy at Birth']:.2f}"
    ]
}
df_table = pd.DataFrame(data)
df_table.set_index('Metrics', inplace=True)
st.table(df_table)
st.markdown("<h2 style='text-align: center;'>Model Performance Summary:</h2>", unsafe_allow_html=True)
st.write(f"R-squared: {r2:.4f}")
st.write(f"Mean Squared Error: {mse:.4f}")
# Display model coefficients
st.markdown("<h2 style='text-align: center;'>Model Coefficients:</h2>", unsafe_allow_html=True)
st.write("Intercept:", f"{model.intercept_:.2f}")
st.write("Coefficient:", f"{model.coef_[0]:.2f}")

st.markdown("<h2 style='text-align: center;'>Data sources:</h2>", unsafe_allow_html=True)
st.write("Data on Life expectancy can be found here from the World Health Organization:https://apps.who.int/gho/data/node.main.688")
st.write("Data on health expenditure can be found here from the World Health Organization:https://apps.who.int/nha/database/Select/Indicators/en")
st.write("Data for universal health care coverage was from multiple sources all cited on the wikipedia page: https://en.wikipedia.org/wiki/Health_care_systems_by_country")
st.write("It should be noted that some data was excluded for the purpose of comparison. Regions not present on all data sets were filtered out.")