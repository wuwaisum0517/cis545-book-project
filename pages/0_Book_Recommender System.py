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
import mysql.connector


def book_recommendation():
    def get_data():
        # AWS_BUCKET_URL = "https://cis5450-project-test.s3.amazonaws.com/"
        # df_books = pd.read_csv(AWS_BUCKET_URL + "cleaned_data_small.csv",encoding="utf-8")
        # Get the current file's directory
        current_dir = os.path.dirname(__file__)

        # Construct the absolute file path
        csv_path = os.path.join(current_dir, "cleaned_data.csv")
        # csv_path = os.path.join(current_dir, "cleaned_data_small.csv") # for the smaller data set testing

        df_books = pd.read_csv(csv_path, encoding="utf-8")
        return df_books

    def mysql_connection():

        """
        Database connection
        """
        mydb = mysql.connector.connect(
            host="cis5450project.cteq6emi8zj5.us-east-1.rds.amazonaws.com",
            user="admin",
            password="cis5450password"
        )
        return mydb

    def execute_sql_query(mydb,query, column_names):
        """
        with mysql_connection information create cursor and return result

        """
        mycursor = mydb.cursor()
        mycursor.execute("USE clean_data;")
        mycursor.execute(query)
        sql_result = mycursor.fetchall()

        dataframe_result = pd.DataFrame(sql_result, columns = column_names)
        mycursor.close()
        return dataframe_result

    def getFullData(mydb):
        """
        Input: cursor from get_data_mysql_connection

        Output: Full COMPLETE data to DataFrame

        """
        query = "SELECT * FROM clean_table "

        column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                        'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                        'Decade-Of-Publication']

        return execute_sql_query(mydb,query, column_names)
    def getTop10Data(mydb):
        """
        Input: cursor from get_data_mysql_connection
        Output: head(10) SQL Data DataFrame
        """
        query = "SELECT * FROM clean_table LIMIT 10 "

        column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                        'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                        'Decade-Of-Publication']

        return execute_sql_query(mydb,query, column_names)
    def getState(mydb):
        """
        Input: cursor from get_data_mysql_connection
        Output: head(10) SQL Data DataFrame
        """
        query = "SELECT * FROM clean_table WHERE State = 'california'"

        column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                        'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                        'Decade-Of-Publication']

        return execute_sql_query(mydb,query, column_names)
    try:
        """
        The starting point
        """
        # connector to SQL server
        mydb = mysql_connection()
        # execute SQL code
        df = getState(mydb)

        isbn_input = st.multiselect(
            "Choose a Book", list(df['ISBN']), ['0440234743']
        )

        if not isbn_input:
            st.error("Please select at least one ISBN.")
        else:
            data = df[['ISBN', 'Book-Title', 'Book-Author']]
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            ax.table(cellText=data.values, colLabels=data.columns, loc='center')
            fig.tight_layout()
            st.pyplot(plt.gcf())  # instead of plt.show()


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
st.write("Section Showing the data the user selected")

st.write("")
st.write("Clicking a button to refresh")

st.write("")
st.write("Section Showing prediction ")

book_recommendation()
show_code(book_recommendation)
