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
import mysqlclient.connector
from sklearn.metrics.pairwise import cosine_similarity

def book_recommendation():
    # def mysql_connection():
    #
    #     """
    #     Database connection
    #     """
    #     mydb = mysql.connector.connect(
    #         host="cis5450project.cteq6emi8zj5.us-east-1.rds.amazonaws.com",
    #         user="admin",
    #         password="cis5450password"
    #     )
    #     return mydb
    def mysql_connection_secret():

        """
        Database connection
        secret variable saved in steamlit
        """
        mydb = mysqlclient.connector.connect(
            host=st.secrets["db_host"],
            user=st.secrets["db_user"],
            password=st.secrets["db_password"]
        )
        st.write('mysql_connection loading completed'+str(dt.datetime.now()))
        return mydb

    def execute_clean_data_sql_query(mydb, query, column_names,debug_mode):
        """
        The database name IS clean_data
        with mysql_connection information create cursor and return result

        """
        mycursor = mydb.cursor()

        mycursor.execute("USE clean_data;")

        if debug_mode:
            st.write(query+' started at '+str(dt.datetime.now()))

        mycursor.execute(query)

        if debug_mode:
            st.write(query+' finished at '+str(dt.datetime.now()))
        sql_result = mycursor.fetchall()
        if debug_mode:
            st.write(query+' started converting DataFrame at '+str(dt.datetime.now()))
        dataframe_result = pd.DataFrame(sql_result, columns=column_names)
        mycursor.close()
        if debug_mode:
            st.write(query+'loading completed at '+str(dt.datetime.now()))
        return dataframe_result

    def get_full_list_book(mydb,debug_mode):
        """
        return df_books
        """
        query = "SELECT * FROM books_table"

        column_names = ['ID', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S',
                        'Image-URL-M', 'Image-URL-L']

        return execute_clean_data_sql_query(mydb, query, column_names,debug_mode)
    def get_book_data_by_title_for_df_books(mydb, book,debug_mode):
        """
        input: mydb and book
        """


        query = "SELECT * FROM books_table WHERE Book_Title = \"{}\"".format(book)

        column_names = ['ID', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S',
                        'Image-URL-M', 'Image-URL-L']
        return execute_clean_data_sql_query(mydb, query, column_names,debug_mode)
    def get_full_list_book_title_data(mydb,debug_mode):
        query = "SELECT DISTINCT Book_Title FROM all_rating"

        column_names = ['Book-Title']

        return execute_clean_data_sql_query(mydb, query, column_names,debug_mode)

    def get_full_list_book_title_data_with_explicit_rating(mydb,debug_mode):
        query = "SELECT DISTINCT Book_Title FROM all_rating WHERE Book_Rating !=0;"

        column_names = ['Book-Title']

        return execute_clean_data_sql_query(mydb, query, column_names,debug_mode)

    def get_book_data_by_title(mydb, book_list,debug_mode):
        """
        Input: cursor from get_data_mysql_connection
        Output: head(10) SQL Data DataFrame
        """
        list_of_books = "\", \"".join(book_list)

        query = "SELECT * FROM clean_table WHERE Book_Title IN (\"{}\")".format(list_of_books)

        column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                        'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                        'Decade-Of-Publication']
        return execute_clean_data_sql_query(mydb, query, column_names,debug_mode)

    def get_explicit_df(mydb,debug_mode):
        """
        UNUSED FUNCTION, for filtering rows
        explicit_df = all_ratings[all_ratings['Book-Rating'] != 0]
        """
        query = "SELECT * FROM all_rating WHERE Book_Rating IS NOT 0"
        column_names = ['User-ID', 'ISBN', 'Book-Title', 'Book-Author', 'Book-Rating']

        return execute_clean_data_sql_query(mydb, query, column_names,debug_mode)

    def get_implicit_df(mydb,debug_mode):
        """
        UNUSED FUNCTION, for filtering rows
        implicit_df = all_ratings[all_ratings['Book-Rating'] == 0]
        """
        query = "SELECT * FROM all_rating WHERE Book_Rating IS 0"
        column_names = ['User-ID', 'ISBN', 'Book-Title', 'Book-Author', 'Book-Rating']

        return execute_clean_data_sql_query(mydb, query, column_names,debug_mode)

    def get_explict_df_matrix(debug_mode):
        """
         This matrix file is generated from :
         explicit_book_user_matrix(explicit_df, 50, 25)
        """
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'explicit_df_matrix.csv')

        loaded_matrix = pd.read_csv(file_path, index_col='Book-Title')
        if debug_mode:
            st.write('matrix loading completed at '+str(dt.datetime.now()))
        book_title_column = pd.read_csv(file_path)
        if debug_mode:
            st.write('column loading completed at '+str(dt.datetime.now()))
        cosine_similarity_calc = cosine_similarity(loaded_matrix)
        if debug_mode:
            st.write('cos similarity calculation loading completed at '+str(dt.datetime.now()))
        return book_title_column['Book-Title'], loaded_matrix, cosine_similarity_calc

    def visualizing_recommendation_data (recommendation_list):
        """
        input: a list of recommendations
        output: code for showing result properly
        """
        for recommendation in recommendation_list:

            #first item
            title, author, image_url, similarity_score = recommendation[0], recommendation[1], recommendation[2], recommendation[3]

            col1, col2 ,col3= st.columns([1,4,2])
            with col1:
                try:
                    st.image(image_url)
                except st.runtime.media_file_storage.MediaFileStorageError:
                    continue
            with col2:
                st.subheader(title)
                st.subheader("Author: "+author)
            with col3:

                st.write("The similarity Score :" +str("{:.2f}".format(similarity_score)))
        return
    def item_based(mydb, explicit_df_matrix, similarity_scores, book_name,debug_mode):
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
            # temp_df = df_books[df_books['Book-Title'] == explicit_df_matrix.index[i[0]]]
            temp_df= get_book_data_by_title_for_df_books(mydb,explicit_df_matrix.index[i[0]],debug_mode)
            rec.extend(temp_df.drop_duplicates('Book-Title')['Book-Title'].to_list())
            rec.extend(temp_df.drop_duplicates('Book-Title')['Book-Author'].to_list())
            rec.extend(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].to_list())
            rec.append(i[1])  # Append the similarity score
            data.append(rec)
        if debug_mode:
            st.write('item_based searched a book '+ book_name +'at'+str(dt.datetime.now()))
        return data


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
        debug_mode = False

        # connector to SQL server
        mydb = mysql_connection_secret()

        default_book_list = ['The Testament', 'Harry Potter and the Goblet of Fire (Book 4)', '1984']
        # BELOW execute SQL query

        # load the precomputed matrix and similarity scoure
        df_title_only, explicit_df_matrix, similarity_scores = get_explict_df_matrix(debug_mode)

        # # loading the title from database
        # df_title_only = get_full_list_book_title_data_with_explicit_rating(mydb)['Book-Title']

        # load the full list fo data , called df_books
        # df_books = get_full_list_book(mydb)

        # multiselect bar

        book_input = st.multiselect(
            "Choose at most 6 book to see our recommendation! ", list(df_title_only), default_book_list
        )

        if not book_input or len(book_input) == 0:
            st.error("Please select at least one Book.")
        else:
            if len(book_input) > 6:
                st.write("Please select at most 5 books only")
                return
            else:
                if debug_mode:
                    st.write('Book_list: ' + (','.join(book_input)))
                for book in book_input:
                    st.write("Because you like \'" + book + "\'...... our recommendation is")
                    if book in st.session_state:
                        recommendation_list = st.session_state[book]
                    else:
                        recommendation_list = (item_based(mydb, explicit_df_matrix, similarity_scores, book,debug_mode))
                        st.session_state[book] = recommendation_list
                    visualizing_recommendation_data(recommendation_list)

                # st.write("DEBUG")
                # st.write("Selected Book: ", book_input)
                # df_target = get_book_data_by_title(mydb, book_input)
                # st.write(df_target)


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
