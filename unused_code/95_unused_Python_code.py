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


def getFullData(mydb):
    """
    Input: cursor from get_data_mysql_connection

    Output: Full COMPLETE data to DataFrame

    """
    query = "SELECT * FROM clean_table "

    column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                    'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                    'Decade-Of-Publication']

    return execute_clean_data_sql_query(mydb, query, column_names)


def getTop10Data(mydb):
    """
    Input: cursor from get_data_mysql_connection
    Output: head(10) SQL Data DataFrame
    """
    query = "SELECT * FROM clean_table LIMIT 10 "

    column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                    'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                    'Decade-Of-Publication']

    return execute_clean_data_sql_query(mydb, query, column_names)


def getState(mydb):
    """
    Input: cursor from get_data_mysql_connection
    Output: head(10) SQL Data DataFrame
    """
    query = "SELECT * FROM clean_table WHERE State = 'california'"

    column_names = ['ID', 'User-ID', 'Age', 'Location', 'City', 'State', 'Country', 'ISBN', 'Book-Rating',
                    'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL', 'User-Decade',
                    'Decade-Of-Publication']

    return execute_clean_data_sql_query(mydb, query, column_names)


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

    return execute_clean_data_sql_query(mydb, query, column_names)


def getISBNOnlyData(mydb):
    query = "SELECT DISTINCT ISBN FROM clean_table"

    column_names = ['ISBN']

    return execute_clean_data_sql_query(mydb, query, column_names)
def book_ratings(mydb):
    """
    FOR EXPLICIT only
    This is the SQL version of the dataframe code below:
    explicit_df = all_ratings[all_ratings['Book-Rating'] != 0]
    book_ratings = explicit_df.groupby('Book-Title').agg({
        'Book-Rating': ['count', 'mean']
    }).reset_index()
    """
    query = """
        SELECT
            Book_Title, COUNT(*),AVG(Book_Rating)
        FROM
            all_rating
        WHERE
            Book_Rating !=0
        GROUP BY
            Book_Title
    """
    column_names = ['Book-Title', 'Number of Ratings', 'Average Rating']
    return execute_clean_data_sql_query(mydb, query, column_names)