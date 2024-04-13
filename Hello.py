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

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        ### Our Team:
        
        Nami Kim
        
        Mason Liebe
        
        Wai-Sum Wu
        
        San Kim

        ### Project Explanation and Detail
        
        
        **Description**  
        This is our CIS 545 Final Project.
        Our intention is to develop a robust methodology for predicting which books a user will like given their previous 
        ratings. The general strategy will be to compute the predicted rating of all the other books in the data set based 
        on the user’s information, then output those with the highest predicted ratings.
        
        ### Technology used:
        AWS
        
        Streamlit
        
        ### Python Package
        numpy
        
        pandas
            
        pydeck

        matplotlib
            
        seaborn
        
        mysql
        
        Data courtesy of the [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/)
        from Kaggle.

    """
    )


if __name__ == "__main__":
    run()
