"""
This is the Book Recommender System Page.
2024-04-09
Working 0%
Testing connection using csv file
SQL Check
"""
from urllib.error import URLError

import altair as alt
import pandas as pd


import streamlit as st
from streamlit.hello.utils import show_code

import numpy as np
import datetime as dt
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm


def book_recommendation():
    def get_data():
        AWS_BUCKET_URL = "https://cis5450-project-test.s3.amazonaws.com/"
        df_books = pd.read_csv(AWS_BUCKET_URL + "cleaned_data_small.csv",encoding="utf-8")
        return df_books

    try:
        df = get_data()
        isbn_input = st.multiselect(
            "Choose a Publisher", list(df.index), [1, 2]
        )

        if not isbn_input:
            st.error("Please select at least one ISBN.")
        else:
            data = df [['ISBN','Book-Title','Book-Author']]
            fig, ax = plt.subplots(figsize = (8,8))
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            ax.table(cellText=data.values,colLabels=data.columns,loc = 'center')
            fig.tight_layout()
            st.pyplot(plt.gcf()) # instead of plt.show()


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

st.markdown("# Book Recommender System")
st.sidebar.header("Book Recommender System")
st.write("")

book_recommendation()
show_code(book_recommendation)
