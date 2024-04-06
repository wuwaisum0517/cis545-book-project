"""
This page for the book recommendation
People Input
expect formatting table

"""
from urllib.error import URLError

import altair as alt
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code
def book_recommendation():

    def get_data():
        AWS_BUCKET_URL = "https://cis5450-project-test.s3.amazonaws.com/"
        df_books = pd.read_csv(AWS_BUCKET_URL + "Books.csv")
        return df_books.set_index("Publisher")

    try:
        df = get_data()
        publisher = st.multiselect(
            "Type Publisher", list(df.index), ["Dell", "Oxford University Press"]
        )
    except URLError as e:
        st.error(
            """
            **AWS connection Problem**
            Connection error: %s
        """
            % e.reason
        )

    return

st.set_page_config(page_title="DataFrame Demo", page_icon="📊")

st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write()
book_recommendation()