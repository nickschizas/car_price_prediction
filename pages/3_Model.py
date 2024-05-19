import streamlit as st
import numpy as np
from scipy.stats import bootstrap

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title='Model', page_icon=':robot_face:')

text = f"""
Model used for prediction is an untuned Random Forest Regressor.<br>
<br>
<b>Model Stats:</b><br>
R-squared: {"{:.2%}".format(st.session_state.model_stats['r2'])}<br>
Mean Absolute Error: {"{:,.2f}".format(st.session_state.model_stats['mae'])}<br>
Residuals Mean: {"{:,.2f}".format(st.session_state.model_stats['res_mean'])}<br>
Residuals Std: {"{:,.2f}".format(st.session_state.model_stats['res_std'])}
"""

st.sidebar.markdown(f'<em>{text}<em>', unsafe_allow_html=True)

residuals = st.session_state.model_stats['residuals']

with st.expander('Confidence Interval of Residual Mean', expanded=False):
  conf_level = st.number_input('Bootstrap Confidence Interval', min_value=.85, max_value=.99, value=.95, step=.01)
  boot = bootstrap((residuals,), np.mean, confidence_level=.99)
  conf_int = boot.confidence_interval
  text = f'{conf_level*100}% confidence interval of residuals mean ({"{:,.2f}".format(conf_int[0])},{"{:,.2f}".format(conf_int[1])}) with standard error {"{:,.2f}".format(boot.standard_error)}'
  st.markdown(text, unsafe_allow_html=True)

fig = make_subplots(rows=1, cols=2, subplot_titles=('Residuals Plot', 'Residuals Histogram'))
fig.add_trace(go.Scatter(x=residuals.index, y=residuals, mode='markers', marker={'color':'#1f77b4', 'opacity':0.6}, hoverinfo='y'), row=1, col=1)
fig.add_hline(y=0, row=1, col=1, line={'color':'red', 'dash':'dot'})
fig.add_trace(go.Histogram(x=residuals, marker={'color':'#1f77b4'}), row=1, col=2)
fig.add_vrect(x0=np.mean(residuals)-np.std(residuals), x1=np.mean(residuals)+np.std(residuals), row=1, col=2,
              annotation_text='68%', annotation_position='top right',
              fillcolor='green', opacity=0.25, line_width=0)

fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(showgrid=False, showline=False, zeroline=False)
fig.update_yaxes(showgrid=False, showline=False, zeroline=False)

st.plotly_chart(fig, use_container_width=True)
