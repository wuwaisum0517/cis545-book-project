from urllib.error import URLError

import altair as alt

import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code

import numpy as np
import datetime as dt
import re
import os
import joblib

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
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
        result_df = result_df.set_axis(column_names, axis='columns')
        if debug_mode:
            st.write(query + ' column name converted ' + str(dt.datetime.now()))
        return result_df

    def loading_user_df (mydb, debug_mode):
        query = 'SELECT * FROM user_df_from_cluster_df'
        column_names = ['User-ID','Age', 'State', 'Book-Rating', 'pages', 'Year-Of-Publication']
        if debug_mode:
            st.write("start execute sql query")
        load = execute_clean_data_sql_query(mydb,query,column_names,debug_mode)

        return load

    def loading_state(mydb, debug_mode):
        query = 'SELECT Distinct State FROM user_df_from_cluster_df ORDER BY State'
        column_names = ['State']
        if debug_mode:
            st.write("start execute sql query")
        load = execute_clean_data_sql_query(mydb,query,column_names,debug_mode)
        return load

    def loading_csv_file(filename):
        """
        Input file name, Output dataframe output
        """
        # Get the current file's directory
        current_dir = os.path.dirname(__file__)

        # Construct the absolute file path
        csv_path = os.path.join(current_dir, str(filename))

        return pd.read_csv(csv_path, encoding="utf-8")

    def generate_python_pickle(users_df, debug_mode):

        """
        dump the TSNE data to pickle file
        """
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

        # Extracting the centroids
        if debug_mode:
            st.write("6 centroids at " + str(dt.datetime.now()))
        centroids = kmeans.cluster_centers_
        if debug_mode:
            st.write("7 centroids done at " + str(dt.datetime.now()))
        # Getting feature names after transformation
        feature_names = numeric_features + list(
            pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())
        if debug_mode:
            st.write("8 feature_names done at " + str(dt.datetime.now()))
        # Creating a DataFrame for the centroids
        centroids_df = pd.DataFrame(centroids, columns=feature_names)
        if debug_mode:
            st.write("9 centroids_df done at " + str(dt.datetime.now()))
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=400)
        if debug_mode:
            st.write("10 TSNE done at " + str(dt.datetime.now()))
        X_tsne = tsne.fit_transform(X_processed)
        if debug_mode:
            st.write("11 X_TSNE done at " + str(dt.datetime.now()))

        # joblib generator
        filename = 'X_tsne.sav'
        joblib.dump(X_tsne, filename)





    def pipeline_process(users_df, new_user_data,debug_mode):
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

        if debug_mode:
            st.write("8 feature_names done at " + str(dt.datetime.now()))

        if debug_mode:
            st.write("9 centroids_df done at " + str(dt.datetime.now()))
        # Apply t-SNE for dimensionality reduction
        # preloaded
        if debug_mode:
            st.write("10 TSNE done at " + str(dt.datetime.now()))

        # X_tsne = joblib.load('X_tsne.sav')

        new_user_df = pd.DataFrame(new_user_data)
        if debug_mode:
            st.write("12 preprocessed_new_user start at " + str(dt.datetime.now()))
        preprocessed_new_user = pipeline.named_steps['preprocessor'].transform(new_user_df).toarray()  # covert sparse matrix to dense array
        if debug_mode:
            st.write("13 dense array done, start predict at " + str(dt.datetime.now()))
        predicted_cluster = kmeans.predict(preprocessed_new_user)
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
        filtered_df = users_df[users_df['User-ID'].isin(similar_user_ids)]

        # Sort the filtered DataFrame by 'Rating' in descending order
        sorted_df = filtered_df.sort_values(by='Book-Rating', ascending=False)

        # Get the top 5 highest-rated books
        # need fix
        top_books = sorted_df.head(5)
        if debug_mode:
            st.write("16: final result")
            st.write(top_books)


        return top_books


    try:
        debug_mode = True

        # connector to SQL server
        mydb = mysql_connection_secret()


        st.write('try this by giving me your information!')
        st.write('Our system will recommend some book for you to check out!')

        # User age?
        # 0-100
        age_input = st.number_input("What is your age? (0-100)",min_value=0,max_value=100)
        requested_age = [age_input]
        options = loading_state(mydb,debug_mode)['State']
        # User state
        state_input = st.multiselect("Which state you comes from?",options)
        requested_state = [state_input]  # Example location
        # User rating
        rating_input = st.number_input("Book Rating - from 0-10",min_value=0,max_value=10)
        requested_rating = [rating_input]
        # User requested pages
        number_of_pages_input = st.number_input("Book Rating - from 0-3800",min_value=0,max_value=3800)       
        requested_pages=[number_of_pages_input]
        # User requested publication year
        year_publication_input = st.number_input("Book Rating - from 0-3800",min_value=1950,max_value=2005)
        requested_year_Of_Publication=[year_publication_input]
        new_user_data = {
            'Age': requested_age,  # Example age
            'State': requested_state,  # Example location
            'Book-Rating': requested_rating,
            'pages': requested_pages,
            'Year-Of-Publication': requested_year_Of_Publication
        }
        if debug_mode:
            st.write(new_user_data)

        st.button("Surprise me !")
        users_df = loading_user_df(mydb,debug_mode)
        pipeline_process(users_df, new_user_data,debug_mode)

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

book_recommendation_based_on_parameter()
show_code(book_recommendation_based_on_parameter)
