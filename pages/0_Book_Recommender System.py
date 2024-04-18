"""
This is the Book Recommender System Page.
2024-04-09
Working 50%
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

    def execute_clean_data_sql_query(mydb, query, column_names):
        """
        The database name IS clean_data
        with mysql_connection information create cursor and return result

        """
        mycursor = mydb.cursor()
        mycursor.execute("USE clean_data;")
        mycursor.execute(query)
        sql_result = mycursor.fetchall()

        dataframe_result = pd.DataFrame(sql_result, columns=column_names)
        mycursor.close()
        return dataframe_result

    def get_full_list_book(mydb):
        """
        return df_books
        """
        query = "SELECT * FROM books_table"

        column_names = ['ID', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S',
                        'Image-URL-M', 'Image-URL-L']

        return execute_clean_data_sql_query(mydb, query, column_names)

    def get_full_list_book_title_data(mydb):
        query = "SELECT DISTINCT Book_Title FROM all_rating"

        column_names = ['Book-Title']

        return execute_clean_data_sql_query(mydb, query, column_names)

    def get_full_list_book_title_data_with_explicit_rating(mydb):
        query = "SELECT DISTINCT Book_Title FROM all_rating WHERE Book_Rating !=0;"

        column_names = ['Book-Title']

        return execute_clean_data_sql_query(mydb, query, column_names)

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
        return execute_clean_data_sql_query(mydb, query, column_names)

    def get_explicit_df(mydb):
        """
        UNUSED FUNCTION, for filtering rows
        explicit_df = all_ratings[all_ratings['Book-Rating'] != 0]
        """
        query = "SELECT * FROM all_rating WHERE Book_Rating IS NOT 0"
        column_names = ['User-ID', 'ISBN', 'Book-Title', 'Book-Author', 'Book-Rating']

        return execute_clean_data_sql_query(mydb, query, column_names)

    def get_implicit_df(mydb):
        """
        UNUSED FUNCTION, for filtering rows
        implicit_df = all_ratings[all_ratings['Book-Rating'] == 0]
        """
        query = "SELECT * FROM all_rating WHERE Book_Rating IS 0"
        column_names = ['User-ID', 'ISBN', 'Book-Title', 'Book-Author', 'Book-Rating']

        return execute_clean_data_sql_query(mydb, query, column_names)

    def get_explict_df_matrix():
        """
         This matrix file is generated from :
         explicit_book_user_matrix(explicit_df, 50, 25)
        """
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'explicit_df_matrix.csv')
        loaded_matrix = pd.read_csv(file_path, index_col='Book-Title')
        return loaded_matrix, cosine_similarity(loaded_matrix)

    def item_based(df_books, explicit_df_matrix, similarity_scores, book_name):
        '''
        uses similarity scores to get an item-based recommendation for a single book title
        Explicit df
        Input: df_books (full list books), get_explict_df_matrix[0],get_explict_df_matrix[1],bookname
        output: calculated data output

        '''
        index = np.where(explicit_df_matrix.index == book_name)[0][0]
        recs = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
        data = []
        for i in recs:
            rec = []
            temp_df = df_books[df_books['Book-Title'] == explicit_df_matrix.index[i[0]]]
            rec.extend(temp_df.drop_duplicates('Book-Title')['Book-Title'].to_list())
            rec.extend(temp_df.drop_duplicates('Book-Title')['Book-Author'].to_list())
            rec.extend(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].to_list())
            rec.append(i[1])  # Append the similarity score
            data.append(rec)
        return data

    def website_data_loading_code(book_list):
        if len(book_list):
            st.write("Please select at most 3 books only")
            return
        else:
            st.write("")
            st.write("Section Showing prediction ")
            for book in book_list:
                st.write("The Name of Book: " + book)
                st.write(item_based(df_books, explicit_df_matrix, similarity_scores, book))

            st.write("")
            st.write("Section Showing the data the user selected")

            st.write("DEBUG")
            st.write("Selected Book: ", book_input)
            df_target = get_book_data_by_title(mydb, book_input)
            st.write(df_target)
        return

        # below the code for plt print the information
        # data = df[['ISBN', 'Book-Title', 'Book-Author']]
        # fig, ax = plt.subplots(figsize=(8, 8))
        # fig.patch.set_visible(False)
        # ax.axis('off')
        # ax.axis('tight')
        # ax.table(cellText=data.values, colLabels=data.columns, loc='center')
        # fig.tight_layout()
        # st.pyplot(plt.gcf())  # instead of plt.show()

    try:
        """
        The starting point of the program
        """

        # connector to SQL server
        mydb = mysql_connection()

        book_list = ['The Testament', 'Harry Potter and the Goblet of Fire (Book 4)', '1984']
        # BELOW execute SQL query

        # loading the title from database
        df_title_only = get_full_list_book_title_data_with_explicit_rating(mydb)
        # load the precomputed matrix and similarity scoure
        explicit_df_matrix, similarity_scores = get_explict_df_matrix()
        # load the full list fo data , called df_books
        df_books = get_full_list_book(mydb)

        # multiselect bar
        book_input = st.multiselect(
            "Choose a Book", list(df_title_only['Book-Title']), options=book_list,
            on_change=website_data_loading_code(book_list)
        )

        if not book_input:
            st.error("Please select at least one Book.")
        else:
            st.write("Section Showing the data the user selected")


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
