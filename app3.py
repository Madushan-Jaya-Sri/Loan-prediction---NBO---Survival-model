# =============================================================================
# import streamlit as st
# import pandas as pd
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from lifelines import CoxPHFitter
# from lifelines.utils import median_survival_times
# 
# # Load the saved CoxPH model
# with open('cox_model.pkl', 'rb') as file:
#     coxph_model = pickle.load(file)
# 
# 
# final_table = pd.read_csv('C:\\Users\\Admin\\final_table.csv')
# final_table.pop('Unnamed: 0')
# 
# 
# # Load the dataset
# df = final_table
# 
# # Define the variables
# var_names = ['Duration', 'status', 'CIFNO', 'LNAMOUNT','LNPAYFREQ', 'LNPERIOD', 'LNINTRATE', 'AVG_BAL_MONTH', 'SAABAL', 'FDGBAL', 'LNODATE', 'LNCLOSEDATE',
#               'CORPORATES', 'FINANCIAL INSTITUTIONS', 'GOVERNMENT INSTITUTIONS',
#               'GOVERNMENT OF SRI LANKA', 'INCORPORATED BODIES', 'INDIVIDUALS', 'LARGE CORPORATES', 'MICRO FINANCE',
#               'MIDDLE MARKET CORPORATES', 'SME', 'SME - BUSINESS BANKING UNDER  SME', 'SME - DEV', 'SME - DEV - REGION',
#               'SME - REGION', 'SME - VERY LARGE', 'STAFF', 'UNCLASSIFIED']
# 
# # Create a list of CIFNOs
# cifnos = df['CIFNO'].unique().tolist()
# st.title("Loan Prediction for NBO Model \n -Survival Analysis-")
# 
# # Create the sidebar
# st.sidebar.title("Customer Selection")
# selected_cifno = st.sidebar.selectbox("Select a CIFNO", cifnos)
# 
# # Filter the dataframe to get the selected customer data
# selected_customer_data = df[df['CIFNO'] == selected_cifno]
# 
# # Prepare the data for the CoxPH model
# selected_customer_data['status'] = selected_customer_data['status'].astype(int)
# selected_customer_data = selected_customer_data[var_names]
# selected_customer_data = selected_customer_data.drop(['status', 'CIFNO'], axis=1)
# 
# # Predict the survival function for the selected customer
# predicted_sf = coxph_model.predict_survival_function(selected_customer_data)
# 
# # Add a slider widget to allow the user to set the x-axis limit manually
# max_time = int(df['Duration'].max())
# x_axis_limit = st.sidebar.slider('Enter the No. of days: \n (future duration) ', 0, max_time, max_time)
# 
# # Plot the survival function with a transparent background and adjusted style for dark mode
# fig, ax = plt.subplots(facecolor='none')
# ax.plot(predicted_sf.index, predicted_sf.values, label=f'CIFNO: {selected_cifno}')
# ax.set_title(f'Survival Function for CIFNO {selected_cifno}', color='white')
# ax.set_xlabel('Time (days)', color='white')
# ax.set_ylabel('Survival probability', color='white')
# ax.set_xlim(0, x_axis_limit)
# ax.legend()
# ax.tick_params(colors='white')
# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white')
# ax.spines['right'].set_color('white')
# ax.spines['left'].set_color('white')
# plt.xticks(color='white')
# plt.yticks(color='white')
# st.pyplot(fig, transparent=True)
# 
# 
# st.markdown("Model Accuracy: 80.2%")
# 
# # =============================================================================
# # for new customers
# # =============================================================================
# print("The number of variables used: ", len(var_names))
# 
# 
# =============================================================================

# =============================================================================
# 
# import streamlit as st
# import pandas as pd
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from lifelines import CoxPHFitter
# from lifelines.utils import median_survival_times
# 
# # Load the saved CoxPH model
# with open('cox_model.pkl', 'rb') as file:
#     coxph_model = pickle.load(file)
# 
# final_table = pd.read_csv('C:\\Users\\Admin\\final_table.csv')
# final_table.pop('Unnamed: 0')
# 
# # Load the dataset
# df = final_table
# 
# # Define the variables
# var_names = ['Duration', 'status', 'CIFNO', 'LNAMOUNT','LNPAYFREQ', 'LNPERIOD', 'LNINTRATE', 'AVG_BAL_MONTH', 'SAABAL', 'FDGBAL', 'LNODATE', 'LNCLOSEDATE',
#               'CORPORATES', 'FINANCIAL INSTITUTIONS', 'GOVERNMENT INSTITUTIONS',
#               'GOVERNMENT OF SRI LANKA', 'INCORPORATED BODIES', 'INDIVIDUALS', 'LARGE CORPORATES', 'MICRO FINANCE',
#               'MIDDLE MARKET CORPORATES', 'SME', 'SME - BUSINESS BANKING UNDER  SME', 'SME - DEV', 'SME - DEV - REGION',
#               'SME - REGION', 'SME - VERY LARGE', 'STAFF', 'UNCLASSIFIED']
# 
# # Create a list of CIFNOs
# cifnos = df['CIFNO'].unique().tolist()
# st.set_page_config(layout="wide")  # Set the page layout to wide
# 
# # Set the title and sidebar
# st.title("Loan Prediction for NBO Model - Survival Analysis")
# st.sidebar.title("Customer Selection")
# selected_cifno = st.sidebar.selectbox("Select a CIFNO", cifnos)
# 
# # Filter the dataframe to get the selected customer data
# selected_customer_data = df[df['CIFNO'] == selected_cifno]
# 
# # Prepare the data for the CoxPH model
# selected_customer_data['status'] = selected_customer_data['status'].astype(int)
# selected_customer_data = selected_customer_data[var_names]
# selected_customer_data = selected_customer_data.drop(['status', 'CIFNO'], axis=1)
# 
# # Predict the survival function for the selected customer
# predicted_sf = coxph_model.predict_survival_function(selected_customer_data)
# 
# # Convert the time values to integers
# predicted_sf.index = predicted_sf.index.astype(int)
# 
# # Add a slider widget to allow the user to set the x-axis limit manually
# max_time = int(df['Duration'].max())
# x_axis_limit = st.sidebar.slider('Enter the No. of days: (future duration)', 0, max_time, max_time)
# 
# # Set the layout to have two columns
# col_graph, col_info = st.columns(2)
# 
# # Plot the survival function with a transparent background and adjusted style for dark mode
# with col_graph:
#     fig, ax = plt.subplots(facecolor='none')
#     ax.plot(predicted_sf.index, predicted_sf.values, label=f'CIFNO: {selected_cifno}')
#     ax.set_title(f'Survival Function for CIFNO {selected_cifno}', color='white')
#     ax.set_xlabel('Time (days)', color='white')
#     ax.set_ylabel('Survival probability', color='white')
#     ax.set_xlim(0, x_axis_limit)
#     ax.legend()
#     ax.tick_params(colors='white')
#     ax.spines['bottom'].set_color('white')
#     ax.spines['top'].set_color('white')
#     ax.spines['right'].set_color('white')
#     ax.spines['left'].set_color('white')
#     plt.xticks(color='white')
#     plt.yticks(color='white')
#     st.pyplot(fig, transparent=True)
# 
# # Display the model accuracy
# with col_info:
#     st.markdown("Model Accuracy: 80.2%")
# 
#     # Display the selected customer data in a table
#     st.subheader("Selected Customer Data")
#     st.table(selected_customer_data)
# 
#     # Manual input for the number of days
#     manual_days = st.number_input('Enter the number of days:', min_value=0, max_value=max_time, value=max_time, step=1)
# 
# # Calculate the survival probability for the given number of days
# if manual_days <= max_time:
#     survival_prob = predicted_sf.loc[manual_days].values[0]
#     survival_prob = 1 - survival_prob
# 
#     # Set the background color based on the survival probability value
#     if survival_prob > 0.75:
#         background_color = 'green'
#     else:
#         background_color = 'black'
# 
#     # Create the HTML code with dynamic background color
#     card_html = f'<div style="background-color: {background_color}; padding: 10px; font-weight: bold; color: white;"> The Probability of getting a loan at the end of {manual_days} days: {survival_prob}</div>'
#     st.markdown(card_html, unsafe_allow_html=True)
# else:
#     st.write("Invalid input. Please enter a valid number of days.")
# 
# st.write("The number of variables used: ", len(var_names)-18)
# 
# =============================================================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times

# Load the saved CoxPH modelddd
with open('cox_model.pkl', 'rb') as file:
    coxph_model = pickle.load(file)

final_table = pd.read_csv('C:\\Users\\Admin\\final_table.csv')
final_table.pop('Unnamed: 0')

# Load the dataset
df = final_table

# Define the variables
var_names = ['Duration', 'status', 'CIFNO', 'LNAMOUNT','LNPAYFREQ', 'LNPERIOD', 'LNINTRATE', 'AVG_BAL_MONTH', 'SAABAL', 'FDGBAL', 'LNODATE', 'LNCLOSEDATE',
              'CORPORATES', 'FINANCIAL INSTITUTIONS', 'GOVERNMENT INSTITUTIONS',
              'GOVERNMENT OF SRI LANKA', 'INCORPORATED BODIES', 'INDIVIDUALS', 'LARGE CORPORATES', 'MICRO FINANCE',
              'MIDDLE MARKET CORPORATES', 'SME', 'SME - BUSINESS BANKING UNDER  SME', 'SME - DEV', 'SME - DEV - REGION',
              'SME - REGION', 'SME - VERY LARGE', 'STAFF', 'UNCLASSIFIED']

# Create a list of CIFNOs
cifnos = df['CIFNO'].unique().tolist()
st.set_page_config(layout="wide")  # Set the page layout to wide

# Set the title and sidebar
st.title("Loan Prediction for NBO Model - Survival Analysis")
st.sidebar.title("Customer Selection")
selected_cifno = st.sidebar.selectbox("Select a CIFNO", cifnos)

# Filter the dataframe to get the selected customer data
selected_customer_data = df[df['CIFNO'] == selected_cifno]

# Prepare the data for the CoxPH model
selected_customer_data['status'] = selected_customer_data['status'].astype(int)
selected_customer_data = selected_customer_data[var_names]
selected_customer_data = selected_customer_data.drop(['status', 'CIFNO'], axis=1)

# Predict the survival function for the selected customer
predicted_sf1 = coxph_model.predict_survival_function(selected_customer_data)

# Convert the time values to integers
predicted_sf1.index = predicted_sf1.index.astype(int)

# Add a slider widget to allow the user to set the x-axis limit manually
max_time = int(df['Duration'].max())
x_axis_limit = st.sidebar.slider('Enter the No. of days: (future duration)', 0, max_time, max_time)

# Set the layout to have two columns
col_graph, col_info  = st.columns(2)

# Plot the survival function with a transparent background and adjusted style for dark mode
with col_graph:
    fig, ax = plt.subplots(facecolor='none')
    ax.plot(predicted_sf1.index, predicted_sf1.values, label=f'CIFNO: {selected_cifno}')
    ax.set_title(f'Survival Function for CIFNO {selected_cifno}', color='white')
    ax.set_xlabel('Time (days)', color='white')
    ax.set_ylabel('Survival probability', color='white')
    ax.set_xlim(0, x_axis_limit)
    ax.legend()
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    st.pyplot(fig, transparent=True)
    st.markdown("Model Accuracy: 80.2%")
# Display the model accuracy
with col_info:
    

    # Display the selected customer data in a table
    st.subheader("Selected Customer Data")
    st.table(selected_customer_data)

    # Manual input for the number of days
    manual_days = st.number_input('Enter the number of days:', min_value=0, max_value=max_time, value=max_time, step=1)

# Calculate the survival probability for the given number of days
if manual_days <= max_time:
    predicted_sf1.reset_index(inplace = True)
    
    predicted_sf1 = predicted_sf1.groupby('index').agg({predicted_sf1.columns[1]:'mean'})
    survival_prob1 = predicted_sf1[predicted_sf1.index==manual_days].values[0][0]
    survival_prob1 = 1 - survival_prob1

    # Set the background color based on the survival probability value
    if survival_prob1 > 0.75:
        background_color = 'green'
    else:
        background_color = 'black'

    # Create the HTML code with dynamic background color
    card_html = f'<div style="background-color: {background_color}; padding: 10px; font-weight: bold; color: white;"> The Probability of getting a loan at the end of {manual_days} days: {survival_prob1}</div>'
    st.markdown(card_html, unsafe_allow_html=True)
else:
    st.write("Invalid input. Please enter a valid number of days.")

# Create empty lists to store the CIFNOs of customers who meet the probability threshold
greater_than_90 = []
greater_than_60 = []
greater_than_30 = []

# Calculate the survival probability for each customer at the given time points and check if it exceeds the threshold
for cifno in cifnos[:]:
    # Filter the dataframe for the current customer
    customer_data = df[df['CIFNO'] == cifno]

    # Prepare the data for the CoxPH model
    customer_data['status'] = customer_data['status'].astype(int)
    customer_data = customer_data[var_names]
    customer_data = customer_data.drop(['status', 'CIFNO'], axis=1)

    # Predict the survival function for the customer
    predicted_sf = coxph_model.predict_survival_function(customer_data)
    predicted_sf.reset_index(inplace = True)
    predicted_sf['index'] = predicted_sf['index'].round().astype('int')
    predicted_sf.groupby('index').agg({predicted_sf.columns[1]:'mean'})

    # Calculate the survival probability at the given time points
    if 90 <= max_time:
        survival_prob_90 = predicted_sf[predicted_sf.index==90].values[0][1]
        survival_prob_90 = 1 - survival_prob_90
        if survival_prob_90 > 0.2:
            greater_than_90.append(cifno)
    if 60 <= max_time:
        survival_prob_60 = predicted_sf[predicted_sf.index==60].values[0][1]
        survival_prob_60 = 1 - survival_prob_60
        if survival_prob_60 > 0.2:
            greater_than_60.append(cifno)
    if 30 <= max_time:
        survival_prob_30 = predicted_sf[predicted_sf.index==30].values[0][1]
        survival_prob_30 = 1 - survival_prob_30
        if survival_prob_30 > 0.2:
            greater_than_30.append(cifno)
            
            
st.subheader("Customers with Probability > 0.75")
# Set the layout to have two columns
table_30 , table_60 ,table_90 = st.columns(3)
# Display the list of customers meeting the criteria


with table_90:
    if 90 <= max_time:
        st.write("At the end of 90 days:")
        df_90 = pd.DataFrame(greater_than_90, columns = ['CIF Number'])
        st.table(df_90)
    
    

with table_60:
    if 60 <= max_time:
        st.write("At the end of 60 days:")
        df_60 = pd.DataFrame(greater_than_60, columns = ['CIF Number'])
        st.table(df_60)
        
with table_30:
    if 30 <= max_time:
        st.write("At the end of 30 days:")
        df_30 = pd.DataFrame(greater_than_30, columns = ['CIF Number'])
        st.table(df_30)

st.write("The number of variables used: ", len(var_names)-18)
