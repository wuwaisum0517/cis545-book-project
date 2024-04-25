# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="CIS 545 Group Project Website",
        page_icon="👋",
    )

    st.write("# Welcome to Our CIS 545 Group Project website! 👋")

    st.sidebar.success("Select a book recommendation system above.  The BookSimilarity Model allows a user to provide books that they've liked in the past, and received recommendations based on those books. The BookClusterer model allows a user to imput books and some demographic information, and receive personalized recommendations.")

    st.markdown(
        """
        ### Our Team:
        
        Nami Kim
        
        Mason Liebe
        
        Wai-Sum Wu
        
        San Kim

        ### Project Details
        
        
        **Description**  
        Welcome to the front end of our CIS 545 Group Project! Our project consists of a variety of book recommendation systems, with techniques ranging from item- and user-based collaborative filtering using cosine similarity and neural networks, to a clustering approach which encorporates user's demographic information.

        Here, we've implemented two recommendation systems in this front end deployment: BookSimilarity and BookClusterer. The BookSimilarity model allows a user to input books that they've liked in the past, and receive recommendations based on those books. The BookClusterer model allows a user to input books and some demographic information, and receive personalized recommendations.

        ### Technologies used:
        AWS (MySQL Server)
        
        Streamlit For builiding website, hosting front end website
        
        ### Python Packages
        Data Manipulation: pandas, numpy, pyspark, SQL, csv, sklearn

        Data Visualization: Plotly, Seaborn, MatPlotLib

        Machine Learning: Tensorflow, Keras, Scikit-learn, NLTK, Gensim

    """
    )
    st.markdown("""---""")

    st.markdown(
        """
        Data courtesy of the [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/)
        from Kaggle and Library of Congress [API](https://www.loc.gov/apis/).
        
        """

    )


if __name__ == "__main__":
    run()
