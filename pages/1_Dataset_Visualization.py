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

    """
    Current Second graph
    """

    writing = """
    
    
    """
    st.write(writing)
    # One maximum is crimson color and the rest are steelblue color
    count_state = df_state['Count'][:15]
    cols = ['steelblue' if (x < max(count_state)) else 'crimson' for x in count_state]

    # State plot
    plt.figure()
    sns.set(rc={'figure.figsize': (12, 6)})
    plot_state = sns.barplot(data=df_state[:15], x='State', y='Count', palette=cols)
    plot_state.set_title('Book Sales in Top 15 States')
    plot_state.set_xticklabels(plot_state.get_xticklabels(), rotation=30)
    plot_state.set_xlabel('State')
    plot_state.set_ylabel('Count')
    for i, v in enumerate(count_state):
        plot_state.text(i, v, str(v), ha='center')
    st.pyplot(plt.gcf())  # instead of plt.show()
    return
# Graph 1 : Book Sales in Top 15 States:
def book_sales_top_15_cities(df_city):

    # Two maximums are crimson color and the rest are steelblue color
    writing = """


    """
    st.write(writing)
    plt.figure()
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
    return

def user_age(df_user_decade):
    writing = """


    """
    st.write(writing)
    plt.figure()
    count_user_decade = df_user_decade['Count']
    cols = ['steelblue' if (x < max(count_user_decade)) else 'crimson' for x in count_user_decade]

    # User decade plot
    sns.set(rc={'figure.figsize': (12, 6)})
    sns.set_style('darkgrid', {'grid.color': 'orchid', 'grid.linestyle': 'dashed'})
    plot_user_decade = sns.barplot(data=df_user_decade, x='User-Decade', y='Count', palette=cols)
    plot_user_decade.set_title('User\'s Ages at Book Sale')
    plot_user_decade.set_xticklabels(plot_user_decade.get_xticklabels(), rotation=30)
    plot_user_decade.set_xlabel('Age')
    plot_user_decade.set_ylabel('Count')
    for i, v in enumerate(count_user_decade):
        plot_user_decade.text(i, v, str(v), ha='center')
    st.pyplot(plt.gcf())  # instead of plt.show()
    return

def book_review_count_by_country(df_country):

    """
    1st graph
    """

    plt.figure()
    plt.figure(figsize=(5,5))
    plt.axis('equal')
    plt.pie(df_country['Count'][:7], labels=df_country['Country'][:7], autopct='%.1f%%')
    plt.title('Book Sales in All Countries')
    st.pyplot(plt.gcf())  # instead of plt.show()
    return


# starting point of the website
# Write/print on this
def website_visualization_start():
    """
    The starting website method
    Logic: Load data, plot the graph, explain
    export the Colab df to csv:
    df_user_decade.to_csv('df_user_decade',index = True)
    """
    try:
        st.title("""Part 1: Data for the Kaggle Dataset""")

        # Create a layout with 10 rows and 2 columns
        col1, col2 = st.columns(2)
        with col1:
            writing = """
            
            59.9% of the review data are from United States contain  , and we will focus on that 
            """
            st.write(writing)
        with col2:
            df_country = loading_csv_file("df_country.csv")
            book_review_count_by_country(df_country)

        df_state = loading_csv_file("df_state.csv")  # load the data
        book_review_count_in_top_15_state(df_state)


        df_city =  loading_csv_file("df_city.csv")  # load the data
        book_sales_top_15_cities(df_city)

        df_user_decade = loading_csv_file("df_user_decade.csv")
        user_age(df_user_decade)


    except URLError as e:
        st.error(
            """
            **CSV loading**
            Connection error: %s
        """
            % e.reason
        )
    return




def loading_csv_file (filename):
    """
    Input file name, Output dataframe output
    """
    # AWS_BUCKET_URL = "https://cis5450-project-test.s3.amazonaws.com/"
    # Get the current file's directory
    current_dir = os.path.dirname(__file__)

    # Construct the absolute file path
    csv_path = os.path.join(current_dir, str(filename))

    df_books = pd.read_csv(csv_path, encoding="utf-8")
    return df_books




st.set_page_config(page_title="Data Visualization", page_icon="📊")


st.sidebar.header("Dataset Graph")
st.title("Exploratory Data Analysis")
st.write("This is the section we show the data from our dataset.")
website_visualization_start()
