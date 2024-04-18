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
def rating_count_by_title(df_title_rating):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.axis('equal')
    plt.pie(df_title_rating['Count'][:15], labels=df_title_rating['Book-Title'][:15], autopct='%.1f%%', pctdistance=0.8)
    plt.title('Count (Title)')
    st.pyplot(plt.gcf())  # instead of plt.show()
def avg_rating_by_title(df_title_rating):
    """
    2.5.2 Average rating by Book Title
    """
    plt.figure()
    df_title_rating_avg = df_title_rating[:30].sort_values(by='Average', ascending=False)
    plt.subplot(2, 1, 2)
    plt.axis('equal')
    plt.pie(df_title_rating_avg['Average'][:15], labels=df_title_rating_avg['Book-Title'][:15], autopct='%.1f%%',
            pctdistance=0.8)
    plt.title('Average (Title)')
    st.pyplot(plt.gcf())  # instead of plt.show()
def two_subplot_graph_by_title(df_title_rating,df_title):

    """
    2.5.2 2 Graph
    """
    plt.figure()
    # One maximum is crimson color and the rest are steelblue color
    count_title_rating = df_title_rating['Average'][:30]
    count_title = df_title['Count'][:30]
    cols_title_rating = ['steelblue' if (x < max(count_title_rating)) else 'crimson' for x in count_title_rating]
    cols_title_count = ['steelblue' if (x < max(count_title)) else 'crimson' for x in count_title]

    # 2 subplots
    _, (plot_title_count, plot_title_average) = plt.subplots(ncols=2, figsize=(12, 12), sharey=True,
                                                             gridspec_kw={'wspace': 0})
    sns.set_style('darkgrid', {'grid.color': 'orchid', 'grid.linestyle': 'dashed'})

    # Title average plot
    plot_title_average = sns.barplot(data=df_title_rating[:30], x='Average', y='Book-Title', orient='y',
                                     palette=cols_title_rating, ax=plot_title_average)
    plot_title_average.tick_params(labelright=True, right=True)
    plot_title_average.set_title(' Top 30 States', loc='left')

    # Title count plot
    plot_title_count = sns.barplot(data=df_title_rating[:30], x='Count', y='Book-Title', orient='y',
                                   palette=cols_title_count, ax=plot_title_count)
    plot_title_count.invert_xaxis()
    plot_title_count.tick_params(labelleft=False, left=False)
    plot_title_count.set_ylabel('')
    plot_title_count.set_title('Book Titles in', loc='right')
    st.pyplot(plt.gcf())  # instead of plt.show()

def rating_count_by_author(df_author_rating):
    """
    2.5.1 Pie Chart - Author Rating Count
    """
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.axis('equal')
    plt.pie(df_author_rating['Count'][:15], labels=df_author_rating['Book-Author'][:15], autopct='%.1f%%',
            pctdistance=0.8)
    plt.title('Count (Author)')
    st.pyplot(plt.gcf())  # instead of plt.show()
def avg_rating_by_author(df_author_rating):
    """
    2.5.1 Pie Chart - Author Average Rating
    """
    plt.figure()
    df_author_rating_avg = df_author_rating[:35].sort_values(by='Average', ascending=False)
    plt.subplot(2, 1, 2)
    plt.axis('equal')
    plt.pie(df_author_rating_avg['Average'][:15], labels=df_author_rating_avg['Book-Author'][:15], autopct='%.1f%%',
            pctdistance=0.8)
    plt.title('Average (Author)')
    st.pyplot(plt.gcf())  # instead of plt.show()
def two_subplot_graph(df_author_rating,df_author):
    """
    2.5.1 2 Graph- Author Average Rating
    """
    plt.figure()
    count_author = df_author['Count'][:35]
    # One maximum is crimson color and the rest are steelblue color
    count_author_rating = df_author_rating['Average'][:35]
    cols_author_rating = ['steelblue' if (x < max(count_author_rating)) else 'crimson' for x in count_author_rating]
    cols_author_count = ['steelblue' if (x < max(count_author)) else 'crimson' for x in count_author]

    # 2 subplots
    _, (plot_author_count, plot_author_average) = plt.subplots(ncols=2, figsize=(9, 9), sharey=True,
                                                               gridspec_kw={'wspace': 0})
    sns.set_style('darkgrid', {'grid.color': 'orchid', 'grid.linestyle': 'dashed'})

    # Author average plot
    plot_author_average = sns.barplot(data=df_author_rating[:35], x='Average', y='Book-Author', orient='y',
                                      palette=cols_author_rating, ax=plot_author_average)
    plot_author_average.tick_params(labelright=True, right=True)
    plot_author_average.set_title(' Top 35 States', loc='left')

    # Author count plot
    plot_author_count = sns.barplot(data=df_author_rating[:35], x='Count', y='Book-Author', orient='y',
                                    palette=cols_author_count, ax=plot_author_count)
    plot_author_count.invert_xaxis()
    plot_author_count.tick_params(labelleft=False, left=False)
    plot_author_count.set_ylabel('')
    plot_author_count.set_title('Book Authors in', loc='right')
    st.pyplot(plt.gcf())  # instead of plt.show()
def top_author_written(df_author_title):
    plt.figure()
    # Two maximums are crimson color and the rest are steelblue color
    count_author_title = df_author_title['Count'][:35]
    cols = ['steelblue' if (x < list(count_author_title.nlargest(2))[1]) else 'crimson' for x in count_author_title]

    # Author title plot
    sns.set(rc={'figure.figsize': (5, 5)})
    sns.set_style('darkgrid', {'grid.color': 'orchid', 'grid.linestyle': 'dashed'})
    plot_author_title = sns.barplot(data=df_author_title[:35], x='Book-Author', y='Count', palette=cols)
    plot_author_title.set_title('How Many Books by an Author')
    plot_author_title.set_xticklabels(plot_author_title.get_xticklabels(), rotation=90)
    plot_author_title.set_xlabel('Book Author')
    plot_author_title.set_ylabel('Number of Books')
    st.pyplot(plt.gcf())  # instead of plt.show()

def top_reviewed_author(df_author):
    plt.figure()
    # Two maximums are crimson color and the rest are steelblue color
    count_author = df_author['Count'][:35]
    cols = ['steelblue' if (x < list(count_author.nlargest(2))[1]) else 'crimson' for x in count_author]

    # Author plot
    sns.set(rc={'figure.figsize': (6, 6)})
    sns.set_style('darkgrid', {'grid.color': 'orchid', 'grid.linestyle': 'dashed'})
    plot_author = sns.barplot(data=df_author[:35], x='Book-Author', y='Count', palette=cols)
    plot_author.set_title('Book Authors in Top 35 States')
    plot_author.set_xticklabels(plot_author.get_xticklabels(), rotation=90)
    plot_author.set_xlabel('Book Author')
    plot_author.set_ylabel('Count')
    st.pyplot(plt.gcf())  # instead of plt.show()

def all_state_vs_cali(df_total_cal):
    plt.figure()
    df_total_cal.set_index("Book-Title", inplace = True)
    df_total_cal.plot(kind='bar', figsize=(12, 6), secondary_y='California Count')
    ax0, ax1 = plt.gcf().get_axes()
    ax0.set_ylabel('Total Count')
    ax1.set_ylabel('California Count')
    plt.title('Top 20 Book Titles in All States vs California (Top State)')
    st.pyplot(plt.gcf())  # instead of plt.show()

def decades_of_publication_vs_book_rating(df_pub_rating):
    plt.figure()
    cmap = sns.diverging_palette(243, 7, as_cmap=True)
    sns.set(rc={'figure.figsize': (12, 6)})
    boxplot_rating = sns.boxplot(
        data=df_pub_rating,
        x='Decade-Of-Publication',
        y='Book-Rating',
        hue='Decade-Of-Publication',
        hue_order=[1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
        palette=cmap
    )
    boxplot_rating.set_title('Decades of Publication vs Book Ratings')
    boxplot_rating.set_xticklabels(boxplot_rating.get_xticklabels(), rotation=90)
    boxplot_rating.set_xlabel('Decade')
    boxplot_rating.set_ylabel('Book Rating')
    st.pyplot(plt.gcf())  # instead of plt.show()
    return


def user_ages_vs_book_rating(df_user_rating):
    plt.figure()
    cmap = sns.diverging_palette(243, 7, as_cmap=True)
    sns.set(rc={'figure.figsize': (12, 6)})
    violinplot_rating = sns.violinplot(
        data=df_user_rating,
        x='User-Decade',
        y='Book-Rating',
        hue='User-Decade',
        hue_order=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210,
                   220, 230],
        palette=cmap
    )
    violinplot_rating.set_title('User Ages vs Book Ratings')
    violinplot_rating.set_xticklabels(violinplot_rating.get_xticklabels(), rotation=90)
    violinplot_rating.set_xlabel('Age')
    violinplot_rating.set_ylabel('Book Rating')
    st.pyplot(plt.gcf())  # instead of plt.show()
    return


def publish_to_age(df_user_pub):
    plt.figure()
    cmap = sns.diverging_palette(243, 7, as_cmap=True)

    sns.set(rc={'figure.figsize': (10, 8)})
    matrix_user_pub = sns.heatmap(data=df_user_pub, fmt='g', cmap=cmap)
    matrix_user_pub.set_xlabel('Decade of Publication')
    matrix_user_pub.set_ylabel('User Age')
    matrix_user_pub.set_title('User Ages and Decades of Publication')
    st.pyplot(plt.gcf())  # instead of plt.show()
    return


def book_publication_distribution(df_pub_decade):
    plt.figure()
    # One maximum is crimson color and the rest are steelblue color
    count_pub_decade = df_pub_decade['Count']
    cols = ['steelblue' if (x < max(count_pub_decade)) else 'crimson' for x in count_pub_decade]

    # Publication decade plot
    sns.set(rc={'figure.figsize': (12, 6)})
    sns.set_style('darkgrid', {'grid.color': 'orchid', 'grid.linestyle': 'dashed'})
    plot_pub_decade = sns.barplot(data=df_pub_decade, x='Decade-Of-Publication', y='Count', palette=cols)
    plot_pub_decade.set_title('Decade of Publication')
    plot_pub_decade.set_xticklabels(plot_pub_decade.get_xticklabels(), rotation=30)
    plot_pub_decade.set_xlabel('Decade')
    plot_pub_decade.set_ylabel('Count')
    for i, v in enumerate(count_pub_decade):
        plot_pub_decade.text(i, v, str(v), ha='center')
    st.pyplot(plt.gcf())  # instead of plt.show()
    return


def book_review_count_in_top_15_state(df_state):
    """
    2nd Graph
    """

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
    1 State Count Pie Chart
    """
    plt.figure(figsize=(5, 5))
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
        st.header("""Part 1: Basic EDA for the Kaggle Dataset""")

        ###################################

        sub_header = """
        1. Which country does the review dataset comes from?
        """
        st.subheader(sub_header)

        col1, col2 = st.columns(2)
        with col1:
            writing = """
            59.9% of the reviews are from United States.
            """
            st.write(writing)
        with col2:
            df_country = loading_csv_file("df_country.csv")  # load data
            book_review_count_by_country(df_country)

        ###################################
        sub_header = """
        2.1 Which State left the most reviews?
        """
        st.subheader(sub_header)

        writing = """

        """
        st.write(writing)
        df_state = loading_csv_file("df_state.csv")  # load the data
        book_review_count_in_top_15_state(df_state)

        ###################################
        sub_header = """
        2.2 Which cities left the most reviews?
        """
        st.subheader(sub_header)

        writing = """

        """
        st.write(writing)
        df_city = loading_csv_file("df_city.csv")  # load the data
        book_sales_top_15_cities(df_city)

        ###################################
        sub_header = """
        3. Age who left reviews?
        """
        st.subheader(sub_header)
        writing = """
        
        """
        st.write(writing)

        df_user_decade = loading_csv_file("df_user_decade.csv")
        user_age(df_user_decade)
        ###################################

        sub_header = """
            2.2.2 Publication Decade: 
        """
        writing = """

        In our DataSet most of the books are from 1990's
        """
        st.subheader(sub_header)
        st.write(writing)
        df_pub_decade = loading_csv_file("df_pub_decade.csv")
        book_publication_distribution(df_pub_decade)
        ###################################
        sub_header = """
            2.2.3 Ages vs Decades
        """
        writing = """

        In our DataSet most of the books are from 1990's
        """
        st.subheader(sub_header)
        st.write(writing)
        df_user_pub = loading_csv_file("df_user_pub.csv")
        publish_to_age(df_user_pub)
        ###################################
        sub_header = """
            2.3 Book Ratings
        """
        writing = """
When examining book ratings, it can be observed that individuals in their 90s tend to give favorable ratings. 
 Note that there are no zero ratings included to avoid confusion due to an excessive number of zero ratings.
        """
        st.subheader(sub_header)
        st.write(writing)
        df_user_rating = loading_csv_file("df_user_rating.csv")
        user_ages_vs_book_rating(df_user_rating)
        ###################################
        sub_header = """
                    2.3.2 Book Ratings and Decades of Publication
                """
        writing = """
It can also be observed that people generally tend to give favorable ratings, especially for ratings from the 1900s and 1940s. 
Note that we have chosen to use 'boxplot' (instead of 'violinplot') to focus more intensively on the median value.
                """
        st.subheader(sub_header)
        st.write(writing)
        df_pub_rating = loading_csv_file("df_pub_rating.csv")
        decades_of_publication_vs_book_rating(df_pub_rating)
        ###################################
        sub_header = """
         2.4.1 Top Books
                    """
        writing = """
       In examining the 'Top 20 Book Titles in All States vs California (Top State)', 
       it's clear that the sales order of books in California closely mirrors the overall sales rankings (not exactly the same), 
       highlighting California's substantial impact on the national book sales landscape. 
       Notably, California emerges as the leading state in terms of book sales.
                        """
        st.subheader(sub_header)
        st.write(writing)
        df_total_cal = loading_csv_file("df_total_cal.csv")
        all_state_vs_cali(df_total_cal)
        ###################################
        sub_header = """
         2.4.2 Top Author
                    """
        writing = """
An intriguing observation is the performance of 'Wild Animus' by Rich Shapero (as above, 2.4.1. Top Books). 
Despite Shapero ranking 35th among authors, 'Wild Animus' astonishingly clinched the 1st place in sales.
                        """
        st.subheader(sub_header)
        st.write(writing)
        df_author = loading_csv_file("df_author.csv")
        top_reviewed_author(df_author)
        ###################################
        sub_header = """
2.4.3. How Many Books by an Author
                        """
        writing = """

When delving into the question of which author has published the most books, 
the findings were quite surprising. Contrary to popular belief, Stephen King and Nora Roberts do not hold this title. 
Instead, Ann M. Martin takes the crown with an impressive 317 published books, closely followed by Francine Pascal with 310 books.
                            """
        st.subheader(sub_header)
        st.write(writing)
        df_author = loading_csv_file("df_author_title.csv")
        top_author_written(df_author)
        ###################################
        sub_header = """
    2.5.1. Count Rating vs Average Rating (Author)
                            """
        writing = """
    When delving into the question of which author has published the most books, 
    the findings were quite surprising. Contrary to popular belief, Stephen King and Nora Roberts do not hold this title. 
    Instead, Ann M. Martin takes the crown with an impressive 317 published books, closely followed by Francine Pascal with 310 books.
                                """
        st.subheader(sub_header)
        st.write(writing)
        df_author_rating = loading_csv_file("df_author_rating.csv")
        col1, col2 = st.columns(2)
        with col1:
            rating_count_by_author(df_author_rating)
        with col2:
            avg_rating_by_author(df_author_rating)
        two_subplot_graph(df_author_rating,df_author)


        ###################################
        sub_header = """
    2.5.2. Count Rating vs Average Rating (Title)
                            """
        writing = """

Number One in Ratings (Title):
Focusing on individual titles, 'Wild Animus' again makes a notable appearance in the Count Rating Category, 
underscoring its popularity. 
Meanwhile, 
'Hary Potter and the Chamber of Secrets (Book 2)' secures the top spot in the Average Rating Category, reflecting its acclaim among readers.
                                """
        st.subheader(sub_header)
        st.write(writing)
        df_title_rating = loading_csv_file("df_title_rating.csv")
        df_title = loading_csv_file("df_title.csv")
        rating_count_by_title(df_title_rating)
        avg_rating_by_title(df_title_rating)
        two_subplot_graph_by_title(df_title_rating,df_title)


    except URLError as e:
        st.error(
            """
            **CSV loading**
            Connection error: %s
        """
            % e.reason
        )
    return


def loading_csv_file(filename):
    """
    Input file name, Output dataframe output
    """
    # AWS_BUCKET_URL = "https://cis5450-project-test.s3.amazonaws.com/"
    # Get the current file's directory
    current_dir = os.path.dirname(__file__)

    # Construct the absolute file path
    csv_path = os.path.join(current_dir,str(filename))

    df_books = pd.read_csv(csv_path, encoding="utf-8")
    return df_books


st.set_page_config(page_title="Data Visualization", page_icon="📊")

st.sidebar.header("Dataset Graph")
st.title("Exploratory Data Analysis")
st.write("This is the section we show the data from our dataset.")
website_visualization_start()
