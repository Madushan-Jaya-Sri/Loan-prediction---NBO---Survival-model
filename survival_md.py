import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics.pairwise import cosine_similarity
import datetime

df_ln = pd.read_csv("E:\\HNB\\files\\dwhlacc2212_Random.csv")
len(df_ln['LNACC'].unique())

df_ca = pd.read_csv("E:\\HNB\\files\\dwhcacc_2301_Random.csv")
len(df_ca['CIFNO'].unique())

df_sa = pd.read_csv("E:\\HNB\\files\\dwhsacc_2301_Random.csv")
len(df_sa['CIFNO'].unique())

df_fd = pd.read_csv("E:\\HNB\\files\\dwhfd_2301_Random.csv")
len(df_fd['FDCIFNO'].unique())

df_cus = pd.read_csv("E:\\HNB\\files\\DWHCIF_2301_Random.csv")
len(df_cus['CIFNO'].unique())


df_merged = pd.read_csv("E:\\HNB\\merged_data\\merged_data.csv")
df_merged.drop(['Unnamed: 0'],axis=1,inplace =True)



df_ln['LNODATE']= pd.to_datetime(df_ln['LNODATE'])
df_ln['LNCLOSEDATE']= pd.to_datetime(df_ln['LNCLOSEDATE'])
df_ln['LNLSTPAY']= pd.to_datetime(df_ln['LNLSTPAY'])
df_ln['LNEXPDT']= pd.to_datetime(df_ln['LNLSTPAY'])


# =============================================================================
# feature engineering
# =============================================================================

df_ln = df_ln[['LNACC','LNAMOUNT','LNINTRATE','LNPERIOD','LNSTATUS','LNPAYFREQ','LNLSTPAY','LNODATE','LNCLOSEDATE',
               'QSPURPOSEDES','CIFNO','LNBASELDESC','LNEXPDT']]


cor_df= df_ln.corr()
sns.heatmap(cor_df, annot = True)

# =============================================================================
# VIF values for variable reduction based on multicollinearity
# =============================================================================


columns = ['LNAMOUNT','LNINTRATE','LNPERIOD','LNPAYFREQ']
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(df_ln[columns].dropna().values, i) for i in range(len(columns))]
vif['Features'] = columns

df_ln.isna().sum()



# =============================================================================
# grouping datasets
# =============================================================================
df_ca.columns
currents_grouped = df_ca.groupby(['CIFNO'],as_index = False).agg({'AVG_BAL_MONTH':'mean'})

df_sa.columns
savings_grouped = df_sa.groupby(['CIFNO'],as_index = False).agg({'SAABAL':'mean'})

df_fd.columns
df_fd.rename(columns = {'FDCIFNO':'CIFNO'},inplace = True)
fixed_grouped = df_fd.groupby(['CIFNO'],as_index = False).agg({'FDGBAL':'mean'})


df_ln_grp= df_ln.groupby('CIFNO').agg({'LNAMOUNT':'mean','LNODATE':'max','LNCLOSEDATE':'max'}).reset_index()

df_ln_grp['status']= df_ln_grp['LNCLOSEDATE'].isna().astype(int)
(df_ln_grp['LNODATE']>=df_ln_grp['LNCLOSEDATE']).value_counts()

df_ln_grp['status']=(df_ln_grp['LNCLOSEDATE'].isna() | (df_ln_grp['LNODATE']>=df_ln_grp['LNCLOSEDATE'])).astype(int)
df_ln_grp['status'].value_counts()


# =============================================================================
# extracting data that were not closed untill - 2022-12-31
# =============================================================================
df_loan_not_closed = df_ln_grp[df_ln_grp['LNCLOSEDATE'].isna()].reset_index()

date_str = '2022-12-31'
from datetime import datetime
date_obj = datetime.strptime(date_str, '%Y-%m-%d')

df_loan_not_closed['Current_date']=date_str
df_loan_not_closed['Duration']= (pd.to_datetime(df_loan_not_closed['Current_date'])-pd.to_datetime(df_loan_not_closed['LNODATE'])).dt.days
df_loan_not_closed['Duration']= df_loan_not_closed['Duration']/30

# =============================================================================
# extracting event happened 
# =============================================================================

df_event_happened = df_ln_grp[(df_ln_grp['LNODATE']>=df_ln_grp['LNCLOSEDATE'])].reset_index()
df_event_happened['Duration']=(df_event_happened['LNODATE'] - df_event_happened['LNCLOSEDATE']).dt.days
df_event_happened['Duration']= df_event_happened['Duration']/30
# =============================================================================
# extracting event not happened
# =============================================================================

df_event_not_happened = df_ln_grp[(df_ln_grp['status']==0)].reset_index()
df_event_not_happened['Duration']=(df_event_not_happened['LNCLOSEDATE'] - df_event_not_happened['LNODATE']).dt.days
df_event_not_happened['Duration']= df_event_not_happened['Duration']/30


# =============================================================================
# appending datasets
# =============================================================================

df_loan_not_closed.drop('Current_date',axis = 1, inplace = True)
ap_df = pd.concat([df_loan_not_closed,df_event_happened,df_event_not_happened],axis = 0).reset_index()
ap_df = ap_df.sort_values(by = 'index')

# =============================================================================
# merging datasets
# =============================================================================

df1 = pd.merge(ap_df,currents_grouped,on=['CIFNO'],how='left')
df2 = pd.merge(df1,savings_grouped,on=['CIFNO'],how='left')
df3 = pd.merge(df2,fixed_grouped,on=['CIFNO'],how='left')
df4 = pd.merge(df3,df_cus[['CIFTITLE','CIFNO','CIFBASLEDESC']],on = ['CIFNO'],how='left')


#df4.drop(['LNBASELDESC'],axis=1,inplace= True)
df4.isna().sum()
df4.columns
df4[['FDGBAL','SAABAL','AVG_BAL_MONTH']] =df4[['FDGBAL','SAABAL','AVG_BAL_MONTH']].fillna(0)
df4.dropna(inplace = True)

#df4.to_csv("merged_data.csv")

df4.duplicated().sum()
#df4['LNLSTPAY']=pd.to_datetime(df4['LNLSTPAY'])
df4['LNODATE'] = pd.to_datetime(df4['LNODATE'])
df4['LNCLOSEDATE'] = pd.to_datetime(df4['LNCLOSEDATE'])
df4.columns
final_table = df4.groupby(['CIFNO']).agg({'CIFBASLEDESC':lambda x : x.mode()[0],'LNAMOUNT':'mean','AVG_BAL_MONTH':'mean','SAABAL':'mean','FDGBAL':'mean','LNODATE':'max','LNCLOSEDATE':'max','status':'max','Duration':'mean'}).reset_index()
# =============================================================================
# 
# =============================================================================
ff = pd.get_dummies(final_table['CIFBASLEDESC'],drop_first=True)
final_table= pd.concat([final_table,ff],axis = 1)
final_table.drop('CIFBASLEDESC',axis = 1 ,inplace = True)
dd = final_table[['Duration','status']]
final_table.drop(['Duration','status'],axis = 1, inplace = True)
final_table = pd.concat([dd,final_table],axis = 1)


# =============================================================================
# final_table['Duration']= df4.groupby('CIFNO')['LNODATE'].diff()
# final_table['Duration']= (final_table['Duration']/np.timedelta64(1,'h'))/24
# final_table.columns
# =============================================================================
# =============================================================================
# 
# num_loan_by_customer = df4.groupby('CIFNO')['LNACC'].nunique().reset_index()
# num_loan_by_customer.rename(columns={'LNACC':'Num_loans'}, inplace =True)
# 
# =============================================================================


from lifelines import CoxPHFitter

# =============================================================================
# removing records which have equal dates of loan open and closed dates. 
# =============================================================================

final_table = final_table[final_table['Duration']!=0.0]
copy_final_table = final_table.copy()

# =============================================================================
# model fitting
# =============================================================================
copy_final_table.drop(['LNODATE', 'LNCLOSEDATE','CIFNO','FDGBAL'],axis = 1 ,inplace =True)

final_table.columns
cph = CoxPHFitter() 
#cph.fit(final_table, duration_col='Duration',event_col='status', formula= ('LNAMOUNT + AVG_BAL_MONTH + SAABAL + FDGBAL  + CIFBASLEDESC'))
cph.fit(copy_final_table, 'Duration', 'status',show_progress=True)
cph.print_summary()
cph.plot()

 # =============================================================================
# getting survival curves for random customers
# =============================================================================

# =============================================================================
# customers = ['998000000219','998R00263650','998R00272211','998000000409']
# 
# 
# selected_customers = final_table[final_table['CIFNO'].isin(customers)]
# selected_customers.columns
# 
# selected_customers = final_table[final_table['CIFNO'].isin(customers)][['Duration','status']]
# selected_customers = final_table[final_table['CIFNO'].isin(customers)][['CIFNO', 'CIFBASLEDESC', 'LNAMOUNT', 'AVG_BAL_MONTH', 'SAABAL',
#        'FDGBAL', 'LNODATE', 'LNCLOSEDATE']]
# 
# cph.predict_survival_function(selected_customers).plot()
# final_table['Duration'].max()
# =============================================================================

customers = ['998000000219','998R00263650','998R00272211','998000000409']
se_rw = final_table[final_table['CIFNO'].isin(customers)].iloc[:,2:]
se_rw = final_table.iloc[20001:20005,2:]


cph.predict_survival_function(se_rw).plot()
plt.xlim(0,90)
plt.xlabel('DAYS')
plt.ylabel('Probability')
plt.title('Cox-Proportional Hazard Regression')

# =============================================================================
# save the model
# =============================================================================
import pickle
with open ('C:\\Users\\Admin\\cox_model.pkl','wb') as f:
    pickle.dump(cph,f)

final_table.to_csv('C:\\Users\\Admin\\final_table.csv')


# =============================================================================
# interactive graph presentation
# =============================================================================
# =============================================================================
# create interactive survival plot for each customer
import plotly.express as px

fig = px.line(final_table, x='Duration', y='CIFNO', color='CIFNO',
              title='Survival Curve for Each Customer')
for i in range(len(fig.data)):
    customer_data = final_table[final_table['CIFNO']==fig.data[i]['name']]
    customer_cph = cph.predict_survival_function(customer_data)
    fig.add_scatter(x=customer_cph.index, y=customer_cph.values, 
                    line=dict(width=2), showlegend=False, 
                    hovertemplate='Customer: ' + str(fig.data[i]['name']) +
                                  '<br>Survival Probability: %{y:.2f}' +
                                  '<br>Duration: %{x:.2f}')
    
fig.show()
# =============================================================================

# =============================================================================
# 
# =============================================================================


import streamlit as st
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times



# Create a list of unique customer IDs
customer_ids = final_table['CIFNO'].unique()

# Define the app layout
st.set_page_config(page_title='Survival Analysis App')
st.title('Survival Analysis App')
st.sidebar.title('Select Customer')
customer_id = st.sidebar.selectbox('Customer ID', customer_ids)

# =============================================================================
# # Filter the data for the selected customer
# customer_data = final_table[final_table['CIFNO'] == customer_id]
# 
# # Fit a Cox proportional hazards model to the customer data
# cph = CoxPHFitter()
# cph.fit(customer_data, duration_col='Duration', event_col='status')
# 
# # Calculate the median survival time for the customer
# median_survival_time = median_survival_times(cph.predict_median(customer_data))[0]
# 
# # Create a survival plot for the customer
# st.plotly_chart(cph.plot_partial_effects_on_outcome('time', [median_survival_time - 1, median_survival_time, median_survival_time + 1]))
# 
# =============================================================================
# Show the survival function for the customer
st.plotly_chart(cph.predict_survival_function(se_rw).plot())







# =============================================================================
# intera
# =============================================================================

import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure, curdoc

# Load data
data = final_table

# Define function to update plot based on selected customer
def update_plot(attr, old, new):
    selected_customer = dropdown.value
    filtered_data = data[data["CIFNO"] == selected_customer]
    source.data = ColumnDataSource.from_df(filtered_data)

# Create plot
source = ColumnDataSource(data)
p = figure(x_axis_label="Time", y_axis_label="Hazard Ratio", plot_width=800, plot_height=400)
p.line(x="time", y="hazard_ratio", source=source)

# Create dropdown to select customer
dropdown = Select(title="Select Customer", value=data["CIFNO"].iloc[0], options=list(data["CIFNO"].unique()))
dropdown.on_change("value", update_plot)

# Create layout and add to document
layout = column(dropdown, p)
curdoc().add_root(layout)
