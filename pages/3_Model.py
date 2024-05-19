import streamlit as st

st.set_page_config(layout="wide", page_title='Model', page_icon=':robot_face:')

st.session_state.model_stats

metric_col = st.columns(4)
with metric_col[0]:
  st.metric(label='R-squared', value="{:.2%}".format(st.session_state.model_stats['r2']))
with metric_col[1]:
  st.metric(label='Mean Absolute Error', value="{:,.2f}".format(st.session_state.model_stats['mae']))
with metric_col[2]:
  st.metric(label='Residuals Mean', value="{:,.2f}".format(st.session_state.model_stats['res_mean']))
with metric_col[0]:
  st.metric(label='Residuals Standard Deviation', value="{:,.2f}".format(st.session_state.model_stats['res_std']))

