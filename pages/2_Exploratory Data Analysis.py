import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title='EDA', page_icon=':chart_with_upwards_trend:')

# Import data
@st.cache_resource
def load_data(data_name):
    data_path = r'./data/'
    return pd.read_csv(data_path+data_name, sep=';')

data = load_data('data_clean_20240509.csv')
# Shape of the data
shape = data.shape
st.sidebar.markdown(f'**Data shape:**<br>*{data.shape[0]:,.0f} rows of data*<br>*{data.shape[1]:,.0f} attributes*', unsafe_allow_html=True)

# Top-n car brands with respect to ads count
top_n = st.sidebar.number_input('top-n', min_value=3, max_value=10, step=1)
ads_counts = data['Name'].value_counts().head(top_n).reset_index()
ads_counts.columns = ['Brand', 'Ads Counts']
st.sidebar.dataframe(ads_counts, use_container_width=True, hide_index=True)

# Filter the data
variable = st.selectbox('Histogram variable', options=['Price', 'Klm', 'CubicCentimetres', 'Horsepower', 'Age'])
filter_col = st.columns(4)
with filter_col[0]:
    name_options = ['All'] + list(data['Name'].unique())
    name = st.selectbox('Brand', options=name_options)
with filter_col[1]:
    gas_type_options = ['All'] + list(data['GasType'].unique())
    gas_type = st.selectbox('Gas Type', options=gas_type_options)
with filter_col[2]:
    gear_box_types = ['All'] + list(data['GearBox'].unique())
    gear_box = st.selectbox('Gear Box Type', options=gear_box_types)
with filter_col[3]:
    percentile = st.number_input('Percentile (0-100)', min_value=0, max_value=100, value=100)
    percentile_price = np.percentile(data[variable], percentile)

# Main columns
main_col = st.columns(2)

# Histograms
with main_col[0]:
    hist_data = data[data[variable]<=percentile_price]
    hist_data = hist_data[hist_data['Name']==name] if name != 'All' else hist_data
    hist_data = hist_data[hist_data['GearBox']==gear_box] if gear_box != 'All' else hist_data
    hist_data = hist_data[hist_data['GasType']==gas_type] if gas_type != 'All' else hist_data

    hist = go.Figure()
    hist.add_trace(go.Histogram(x=hist_data[variable], marker_color='lightgreen'))

    hist.update_layout(title_text=f'Histogram of the {variable}',
                       title_font_size=20, xaxis_title = variable, yaxis_title = 'count',
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(hist,use_container_width=True)

# Scatterplots
with main_col[1]:
    scatter_data = data[data[variable]<=percentile_price]
    scatter_data = scatter_data[scatter_data['Name']==name] if name != 'All' else scatter_data
    scatter_data = scatter_data[scatter_data['GearBox']==gear_box] if gear_box != 'All' else scatter_data
    scatter_data = scatter_data[scatter_data['GasType']==gas_type] if gas_type != 'All' else scatter_data

    scatter = go.Figure()
    for gastype in scatter_data['GasType'].unique():
        scatter_data_gastype = scatter_data[scatter_data['GasType']==gastype]
        scatter.add_trace(go.Scatter(x=scatter_data_gastype[variable], y=scatter_data_gastype['Price'],
                                 mode='markers', marker_opacity=.6, name=gastype))

    scatter.update_layout(title_text=f'{variable} vs Price',title_font_size=20,
                          xaxis_title = variable, yaxis_title = 'Price', legend_title = 'Gas Type',
                          xaxis_range=[0, None], yaxis_range=[0, None],
                          yaxis = dict(showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(scatter,use_container_width=True)

# Pie chart
with main_col[0]:
    top_n = 4
    gastype_counts = data['GasType'].value_counts().reset_index()
    gastype_counts_binned = gastype_counts.loc[:top_n-2]
    gastype_counts_other = gastype_counts.loc[top_n-1:]['count'].sum()
    gastype_counts_binned.loc[len(gastype_counts_binned)] = ('Other', gastype_counts_other)
    gastype_counts_binned['pull'] = np.where(gastype_counts_binned['GasType'] == 'Other', 0.2, 0)
    pie = go.Figure()
    pie.add_trace(go.Pie(labels=gastype_counts_binned['GasType'], values=gastype_counts_binned['count'], pull=gastype_counts_binned['pull'],
                         hoverinfo='label+percent',
                         textposition='inside', textinfo='value', insidetextorientation='radial'))
    pie.update_layout(title_text='Gas Type counts allocation',title_font_size=20,
                      legend_title = 'Gas Type',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(pie, use_container_width=True)

# Boxplots
with main_col[1]:
    box_data = data.copy()
    boxplot = go.Figure()
    for gastype in box_data['GasType'].unique():
        box_data_filter = box_data[box_data['GasType']==gastype]
        boxplot.add_trace(go.Box(y=box_data_filter['Price'], name=gastype))

    boxplot.update_layout(title_text='Gas Type vs Price', title_font_size=20,
                          yaxis_title = 'Price', legend_title = 'Gas Type',
                          yaxis = dict(showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    st.plotly_chart(boxplot,use_container_width=True)