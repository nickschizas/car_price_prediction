import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title='Model', page_icon=':robot_face:')

text = """
Model used for prediction is an untuned Random Forest Regressor.
"""

st.sidebar.markdown(f'<em>{text}<em>', unsafe_allow_html=True)

metric_col = st.columns(4)
with metric_col[0]:
  st.metric(label='R-squared', value="{:.2%}".format(st.session_state.model_stats['r2']))
with metric_col[1]:
  st.metric(label='Mean Absolute Error', value="{:,.2f}".format(st.session_state.model_stats['mae']))
with metric_col[2]:
  st.metric(label='Residuals Mean', value="{:,.2f}".format(st.session_state.model_stats['res_mean']))
with metric_col[3]:
  st.metric(label='Residuals Standard Deviation', value="{:,.2f}".format(st.session_state.model_stats['res_std']))

residuals = st.session_state.model_stats['residuals']
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
