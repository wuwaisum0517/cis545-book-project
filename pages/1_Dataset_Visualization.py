
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



#Graph 1 : Book Sales in Top 15 States:
def book_sales_top_15_states(df_cleaned):
    df_state = df_cleaned[['State', 'Book-Rating']].groupby(by='State', as_index=False).count()
    df_state.rename(columns={'Book-Rating': 'Count'}, inplace=True)
    df_state.sort_values(by='Count', ascending=False, inplace=True)
    count_state = df_state['Count'][:15]
    cols = ['steelblue' if (x < max(count_state)) else 'crimson' for x in count_state]

    # State plot
    sns.set(rc={'figure.figsize': (12, 6)})
    plot_state = sns.barplot(data=df_state[:15], x='State', y='Count', palette=cols)
    plot_state.set_title('Book Sales in Top 15 States')
    plot_state.set_xticklabels(plot_state.get_xticklabels(), rotation=30)
    plot_state.set_xlabel('State')
    plot_state.set_ylabel('Count')
    for i, v in enumerate(count_state):
        plot_state.text(i, v, str(v), ha='center')
    st.pyplot(plt.gcf())  # instead of plt.show()


#starting point of the website
#Write/print on this
def website_visualization_start():
    try:
        df_cleaned = load_data() #load the data
        book_sales_top_15_states(df_cleaned)

    except URLError as e:
        st.error(
            """
            **AWS database connection Problem**
            Connection error: %s
        """
            % e.reason
        )
    return

def load_data():
    # AWS_BUCKET_URL = "https://cis5450-project-test.s3.amazonaws.com/"
    # Get the current file's directory
    current_dir = os.path.dirname(__file__)

    # Construct the absolute file path
    csv_path = os.path.join(current_dir, "cleaned_data.csv")

    df_books = pd.read_csv(csv_path,encoding="utf-8")
    return df_books






website_visualization_start()


st.set_page_config(page_title="Data Visualization", page_icon="📊")


st.markdown("Dataset Graph")
st.sidebar.header("Dataset Graph")
st.write("This is the section we show the data from our dataset.")