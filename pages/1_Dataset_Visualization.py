"""


"""
from urllib.error import URLError

import altair as alt
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code
import json
import glob
import pandas as pd
import numpy as np
import datetime as dt
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from collections import Counter

def book_review_count_in_top_15_state(df_state):

    # One maximum is crimson color and the rest are steelblue color
    count_state = df_state['Count'][:15]
    cols = ['steelblue' if (x < max(count_state)) else 'crimson' for x in count_state]

    # State plot
    plt.clf()
    sns.set(rc={'figure.figsize': (12, 6)})
    plot_state = sns.barplot(data=df_state[:15], x='State', y='Count', palette=cols)
    plot_state.set_title('Book Sales in Top 15 States')
    plot_state.set_xticklabels(plot_state.get_xticklabels(), rotation=30)
    plot_state.set_xlabel('State')
    plot_state.set_ylabel('Count')
    for i, v in enumerate(count_state):
        plot_state.text(i, v, str(v), ha='center')
    st.pyplot(plt.gcf())  # instead of plt.show()
# Graph 1 : Book Sales in Top 15 States:
def book_sales_top_15_cities(df_city):

    # Two maximums are crimson color and the rest are steelblue color
    plt.clf()
    count_city = df_city['Count'][:15]
    cols = ['steelblue' if (x < 6700) else 'crimson' for x in count_city]

    # City plot
    sns.set(rc={'figure.figsize': (12, 6)})
    plot_city = sns.barplot(data=df_city[:15], x='City', y='Count', palette=cols)
    plot_city.set_title('Book Sales in Top 15 Cities')
    plot_city.set_xticklabels(plot_city.get_xticklabels(), rotation=30)
    plot_city.set_xlabel('City')
    plot_city.set_ylabel('Count')
    for i, v in enumerate(count_city):
        plot_city.text(i, v, str(v), ha='center')
    st.pyplot(plt.gcf())  # instead of plt.show()



# starting point of the website
# Write/print on this
def website_visualization_start():
    """
    The starting website method
    website write on this
    """
    try:


        df_state = load_state()  # load the data
        book_review_count_in_top_15_state(df_state)

        df_city = load_cities()  # load the data
        book_sales_top_15_cities(df_city)
    except URLError as e:
        st.error(
            """
            **AWS database connection Problem**
            Connection error: %s
        """
            % e.reason
        )
    return


def load_cities():
    # AWS_BUCKET_URL = "https://cis5450-project-test.s3.amazonaws.com/"
    # Get the current file's directory
    current_dir = os.path.dirname(__file__)

    # Construct the absolute file path
    csv_path = os.path.join(current_dir, "df_city.csv")

    df_books = pd.read_csv(csv_path, encoding="utf-8")
    return df_books

def load_state():
    # Get the current file's directory
    current_dir = os.path.dirname(__file__)

    # Construct the absolute file path
    csv_path = os.path.join(current_dir, "df_state.csv")

    df_state = pd.read_csv(csv_path, encoding="utf-8")
    return df_state


st.set_page_config(page_title="Data Visualization", page_icon="📊")
website_visualization_start()
st.markdown("Dataset Graph")
st.sidebar.header("Dataset Graph")
st.write("This is the section we show the data from our dataset.")
