"""
Created on 27/07/2022 16:34
@author: Liu Ziyan
@E-mail: 2650906L@student.gla.ac.uk
"""
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events

# Writes a component similar to st.write()
fig1 = px.line(x=[1], y=[1])
st.session_state['1'] = plotly_events(fig1)

# Can write inside of things using with

# Select other Plotly events by specifying kwargs
fig = px.line(x=[2], y=[2])
st.session_state['2'] = plotly_events(fig)

area = st.empty()

if st.session_state['1']:
    with area.container():
        st.write(st.session_state['1'])
        st.session_state['1'] = []
if st.session_state['2']:
    with area.container():
        st.write(st.session_state['2'])
        st.session_state['2'] = []

