import streamlit as st

st.title("Process to create this website")

st.write("""We """)

st.write("From CSV to AWS S3")

st.write("SQL query handling")


st.write("""
File structure:

df_country.csv -> df_country (1)


""")

st.write("""
source -> SQL Table -> dataframe
###########################
-> df_users_usa = df_users[df_users['Country'] == 'usa']
-> df_combined = df_users_usa.merge(df_ratings, how='inner', on='User-ID').merge(df_books_usa, how='inner', on='ISBN')
-> df_cleaned = df_combined.copy() -> df_cleaned
-> df_cleaned after the graph
-> ***export as CSV***
-> SQL Table: clean_table 
-> end up UNUSED

##################
-> df_ratings + df_books dataset
-> df_users.merge(df_ratings, how='inner', on='User-ID').merge(df_books, how='inner', on='ISBN')
-> *** export as CSV***
SQL Table: all_rating 
-> 
end up UNUSED


##################

df_books dataset : pd.read_csv('/content/Books.csv')

 -> SQL Table: books_table 
 -> df_books 


##################

loc-50-percent.csv from Nami 
-> SQL Table: loc_data 

##################

cluster_df = df_cleaned.merge(loc_clean, left_on='ISBN', right_on='isbn')
users_df = cluster_df['User-ID', 'Age', 'State', 'Book-Rating', 'pages', 'Year-Of-Publication']

users_df = users_df.groupby('User-ID').agg({
  'Age': 'mean',                      # average age
  'State': lambda x: x.mode()[0],     # mode of state
  'Book-Rating': 'mean',              # average age
  'pages': 'mean',                    # average pages of book they read
  'Year-Of-Publication': 'mean'       # average year of publication year
}).reset_index()

-> SQL Table user_df_from_cluster_df 

""")