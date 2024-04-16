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
from sklearn.metrics.pairwise import cosine_similarity

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

    def get_data_by_isbn(mydb, isbn_list):
        """
        Input: cursor from get_data_mysql_connection
        Output: head(10) SQL Data DataFrame
        """
        list_of_isbn = "', '".join(isbn_list)


        query = "SELECT * FROM clean_table WHERE ISBN IN ('{}')".format(list_of_isbn)

        column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                        'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                        'Decade-Of-Publication']

        return execute_sql_query(mydb,query, column_names)

    def getISBNOnlyData(mydb):
        query = "SELECT DISTINCT ISBN FROM clean_table"

        column_names = ['ISBN']

        return execute_sql_query(mydb,query, column_names)

    def get_full_list_book_title_data(mydb):
        query = "SELECT DISTINCT Book_Title FROM clean_table"

        column_names = ['Book-Title']

        return execute_sql_query(mydb,query, column_names)

    def get_book_data_by_title(mydb, book_list):
        """
        Input: cursor from get_data_mysql_connection
        Output: head(10) SQL Data DataFrame
        """
        list_of_books = "\", \"".join(book_list)


        query = "SELECT * FROM clean_table WHERE Book_Title IN (\"{}\")".format(list_of_books)

        column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                        'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                        'Decade-Of-Publication']
        st.write("Debug log:get_book_data_by_title "+query)
        return execute_sql_query(mydb,query, column_names)

    def get_explicit_df(mydb):
        """
        UNUSED FUNCTION, for filtering rows
        explicit_df = all_ratings[all_ratings['Book-Rating'] != 0]
        """
        query = "SELECT * FROM clean_table WHERE Book_Rating IS NOT 0"
        column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                        'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                        'Decade-Of-Publication']

        return execute_sql_query(mydb,query, column_names)

    def get_implicit_df(mydb):
        """
        UNUSED FUNCTION, for filtering rows
        implicit_df = all_ratings[all_ratings['Book-Rating'] == 0]
        """
        query = "SELECT * FROM clean_table WHERE Book_Rating IS 0"
        column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                        'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                        'Decade-Of-Publication']

        return execute_sql_query(mydb, query, column_names)

    def book_rating_rating(mydb):
        """
        The SQL version of the code below:
        explicit_df = all_ratings[all_ratings['Book-Rating'] != 0]
        book_ratings = explicit_df.groupby('Book-Title').agg({
            'Book-Rating': ['count', 'mean']
        }).reset_index()
        """
        query = """
        SELECT  
            Book_Title, COUNT (*),AVG (*)
        FROM 
            Clean_table
        WHERE 
            Book_Rating IS NOT 0
        GROUP BY 
            Book_Title
        """
        column_names = ['Book-Title', 'Number of Ratings', 'Average Rating']
        return execute_sql_query(mydb, query, column_names)

    def get_matrix():
        """
         This matrix file is generated from :
         explicit_book_user_matrix(explicit_df, 50, 25)

        """
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'matrix.csv')
        loaded_matrix = pd.read_csv(file_path)
        return loaded_matrix, cosine_similarity(matrix)


    def item_based(similarity_scores, matrix, book_name):
      '''
      Make a book recommendtaion based on a single book title
      uses similarity scores to get an item-based recommendation for a single book title
      '''
      index = np.where(matrix.index==book_name)[0][0]
      recs = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1], reverse=True)[1:6]
      data = []
      for i in recs:
        rec = []
        temp_df = df_books[df_books['Book-Title'] == matrix.index[i[0]]]
        rec.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        rec.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        rec.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        rec.append(i[1])  # Append the similarity score
        data.append(rec)
      return data

    try:
        """
        The starting point of the program
        """

        matrix = get_matrix()
        # connector to SQL server
        mydb = mysql_connection()

        book_list = ['The Testament','Harry Potter and the Goblet of Fire (Book 4)','1984']
        # execute SQL code

        df_title_only = get_full_list_book_title_data(mydb)

        book_input = st.multiselect(
            "Choose a Book", list(df_title_only['Book-Title']), book_list
        )





        if not book_input:
            st.error("Please select at least one Book.")
        else:
            st.write("Selected Book: ", book_input)
            df_target = get_book_data_by_title(mydb,book_input)

            st.write("")
            st.write("Section Showing the data the user selected")
            st.write(df_target)

            st.write("")
            st.write("Clicking a button to refresh")

            st.write("")
            st.write("Section Showing prediction ")

            # below the code for plt print the information
            # data = df[['ISBN', 'Book-Title', 'Book-Author']]
            # fig, ax = plt.subplots(figsize=(8, 8))
            # fig.patch.set_visible(False)
            # ax.axis('off')
            # ax.axis('tight')
            # ax.table(cellText=data.values, colLabels=data.columns, loc='center')
            # fig.tight_layout()
            # st.pyplot(plt.gcf())  # instead of plt.show()


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


book_recommendation()
show_code(book_recommendation)
