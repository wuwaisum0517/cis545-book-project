from urllib.error import URLError

import altair as alt

import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code

import numpy as np
import datetime as dt


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def book_recommendation_based_on_parameter():
    def mysql_connection_secret():
        """
        Database connection
        """
        mydb = st.connection('mysql', type='sql')
        return mydb

    def execute_clean_data_sql_query(mydb, query, column_names, debug_mode):
        """
        implemented based on official document
        """
        if debug_mode:
            st.write(query + ' started at' + str(dt.datetime.now()))
        result_df = mydb.query(query, ttl=600)
        if debug_mode:
            st.write(query + ' loading completed at ' + str(dt.datetime.now()))
            st.write(result_df)
        result_df = result_df.set_axis(column_names, axis='columns')
        if debug_mode:
            st.write(query + ' column name converted ' + str(dt.datetime.now()))
        return result_df

    def loading_user_df(mydb, debug_mode):
        query = 'SELECT * FROM user_df_from_cluster_df'
        column_names = ['User-ID', 'Age', 'State', 'Book-Rating', 'pages', 'Year-Of-Publication']
        if debug_mode:
            st.write("start execute sql query")
        load = execute_clean_data_sql_query(mydb, query, column_names, debug_mode)

        return load

    def loading_state(mydb, debug_mode):
        query = 'SELECT Distinct State FROM user_df_from_cluster_df ORDER BY State'
        column_names = ['State']
        if debug_mode:
            st.write("start execute sql query")
        load = execute_clean_data_sql_query(mydb, query, column_names, debug_mode)
        return load

    def loading_books_info(mydb, similar_user_ids, debug_mode):
        similar_user_ids_str = (',').join(map(str, similar_user_ids))
        query = 'SELECT * FROM cluster_database where User_ID IN (' + similar_user_ids_str + ')'
        column_names = ['ID', 'User-ID', 'Age', 'State', 'ISBN', 'Title', 'Book-Rating', 'book-author',
                        'Year-of-Publication', 'Publisher', 'pages']
        if debug_mode:
            st.write("start execute sql query " + query)
        load = execute_clean_data_sql_query(mydb, query, column_names, debug_mode)
        return load

    def load_book_title(mydb):
        query = 'SELECT DISTINCT Title FROM cluster_database'
        column_names = ['Title']
        if debug_mode:
            st.write("start execute sql query " + query)
        load = execute_clean_data_sql_query(mydb, query, column_names, debug_mode)
        return load

    def load_book_given_book_title(mydb, book_title, debug_mode):
        query = "SELECT * FROM cluster_database WHERE Title = \"{}\"".format(book_title)

        column_names = ['ID', 'User-ID', 'Age', 'State', 'ISBN', 'Title', 'Book-Rating', 'book-author',
                        'Year-of-Publication', 'Publisher', 'pages']
        if debug_mode:
            st.write("start execute sql query " + query)
        load = execute_clean_data_sql_query(mydb, query, column_names, debug_mode)
        return load

    def try_books_picture(mydb, ISBN, debug_mode):
        query = "SELECT Image_URL_M FROM books_table WHERE ISBN = \'" + str(ISBN) + "\'"

        column_names = ['Image_URL_M']
        load = execute_clean_data_sql_query(mydb, query, column_names, debug_mode)
        if len(load['Image_URL_M']) == 0:
            return ''
        else:
            if debug_mode:
                st.write(load['Image_URL_M'])
            return load['Image_URL_M'][0]

    def pipeline_process(book_input, users_df, new_user_data, debug_mode):
        # Define preprocessing for numeric columns (scaling)
        numeric_features = ['Age', 'Book-Rating', 'pages', 'Year-Of-Publication']
        numeric_transformer = StandardScaler()

        # Define preprocessing for categorical features (one-hot encoding)
        categorical_features = ['State']
        categorical_transformer = OneHotEncoder()

        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create a preprocessing pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

        # Apply transformations
        if debug_mode:
            st.write("1 started X_processed " + str(dt.datetime.now()))
        X_processed = pipeline.fit_transform(users_df).toarray()  # covert sparse matrix to dense array


        if debug_mode:
            st.write("2 finished X_processed , start KMean" + str(dt.datetime.now()))
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        if debug_mode:
            st.write("3 finished KMeans " + str(dt.datetime.now()))
        kmeans.fit(X_processed)
        if debug_mode:
            st.write("4 finished fitting " + str(dt.datetime.now()))
        labels = kmeans.labels_

        if debug_mode:
            st.write("5 clustered_users_df started at " + str(dt.datetime.now()))
        clustered_users_df = users_df.copy()
        clustered_users_df['cluster'] = labels

        new_user_df = pd.DataFrame(new_user_data)
        if debug_mode:
            st.write("12 preprocessed_new_user start at " + str(dt.datetime.now()))
        preprocessed_new_user = pipeline.named_steps['preprocessor'].transform(
            new_user_df).toarray()  # covert sparse matrix to dense array
        if debug_mode:
            st.write("13 dense array done, start predict at " + str(dt.datetime.now()))
        predicted_cluster = kmeans.predict(preprocessed_new_user)
        if debug_mode:
            st.write("14 The new user belongs to cluster:", predicted_cluster[0])

        # Compute distances from the new user to all existing users
        distances = np.linalg.norm(X_processed - preprocessed_new_user, axis=1)

        # Get the indices of the 5 closest users
        closest_indices = np.argsort(distances)[:5]

        selected_rows = clustered_users_df.iloc[closest_indices]
        if debug_mode:
            st.write(selected_rows)

        similar_user_ids = selected_rows['User-ID']

        # Filter the DataFrame for these users
        filtered_df = loading_books_info(mydb, similar_user_ids, debug_mode)

        # Sort the filtered DataFrame by 'Rating' in descending order

        sorted_df = filtered_df.sort_values(by='Book-Rating', ascending=False)
        sorted_df = sorted_df[sorted_df['Title']!= book_input]
        # Remove duplicates based on isbn to ensure uniqueness, keep the highest rating
        unique_sorted_df = sorted_df.drop_duplicates(subset='ISBN', keep='first')
        unique_sorted_df.head(5)

        if debug_mode:
            st.write("16: final result")
            st.write(top_books)
        return unique_sorted_df.head(5).reset_index()

    def showing_result(top_books, debug_mode):
        if debug_mode:
            st.write(top_books)
        for index, row in top_books.iterrows():
            st.subheader(row['Title'])
            st.write('Book Information')
            col0, col1, col2 = st.columns([1, 4, 2])

            with col0:
                st.image(try_books_picture(mydb, row['ISBN'], debug_mode))
            with col1:
                st.write("Author: " + row['book-author'])
                st.write('ISBN: ' + row['ISBN'])
            with col2:
                st.write("Book Page: " + str(int(row['pages'])))
                st.write("Year of Publication: " + str(int(row['Year-of-Publication'])))
                st.write("Book Rating: " + str("{:.2f}".format (row['Book-Rating'])))
            st.markdown("""---""")
        return

    try:
        debug_mode = False

        # connector to SQL server
        mydb = mysql_connection_secret()
        # choice of book
        book_title_list = load_book_title(mydb)
        book_input = st.selectbox("Select a book", book_title_list['Title'])
        load_book_information = load_book_given_book_title(mydb, book_input, debug_mode)

        requested_rating = load_book_information['Book-Rating'][0]
        requested_pages = load_book_information['pages'][0]
        requested_year_of_publication = load_book_information['Year-of-Publication'][0]
        # User age?
        # 0-100
        age_input = st.number_input("Please enter your age in years (0-100)", min_value=0, max_value=100, value=28)
        requested_age = [age_input]
        options = loading_state(mydb, debug_mode)['State']
        # User state
        state_input = st.selectbox("Please enter the state you are from.", options, index=5)
        requested_state = [state_input]  # Example location


        st.markdown("""---""")
        st.subheader("Our recommendation:")
        new_user_data = {
            'Age': requested_age,  # Example age
            'State': requested_state,  # Example location
            'Book-Rating': requested_rating,
            'pages': requested_pages,
            'Year-Of-Publication': requested_year_of_publication
        }
        if debug_mode:
            st.write(new_user_data)

        users_df = loading_user_df(mydb, debug_mode)
        top_books = pipeline_process(book_input, users_df, new_user_data, debug_mode)
        showing_result(top_books, debug_mode)

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
st.markdown("# BookClusterer")
st.sidebar.header("Clustering-based Book Recommender System")

st.subheader('Welcome to our BookClusterer! Please input your information below to receive personalized book recommendations.')
book_recommendation_based_on_parameter()
show_code(book_recommendation_based_on_parameter)
