import streamlit as st

st.set_page_config(layout="wide", page_title='Model', page_icon=':robot_face:')

st.session_state.model_stats

metric_col = st.columns(4)
with metric_col[0]:
  st.metric(label='R-squared', value="{:.2%}".format(st.session_state.model_stats['r2'])
