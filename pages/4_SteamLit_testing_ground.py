"""
Basic testing Only

"""
from urllib.error import URLError

import altair as alt
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code
def testing ():
    slider = st.slider("Testing Slider", min_value = 10, max_value=100)
    select_slider = st.select_slider("Select Slider", options = ['a','b','c'])
    st.write ("Var Slider:"+str(slider))
    st.write ("Var Slider:"+str(select_slider))

    return


st.set_page_config(page_title="Testing", page_icon="📊")
st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write()

testing()
show_code(testing)
