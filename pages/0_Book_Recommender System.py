"""
This is the Book Recommender System Page.
2024-04-09
Working 0%
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
        publisher_input = st.multiselect(
            "Choose a Publisher", list(df.index), ["Dell", "Oxford University Press"]
        )

        if not publisher_input:
            st.error("Please select at least one Publisher.")
        else:
            data = df.loc[publisher_input]
            data = pd.melt(data, id_vars=['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'])


    except URLError as e:
        st.error(
            """
            **AWS database connection Problem**
            Connection error: %s
        """
            % e.reason
        )

    return


st.set_page_config(page_title="Book Recommender System Demo", page_icon="📊")

st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write()

book_recommendation()
show_code(book_recommendation)
