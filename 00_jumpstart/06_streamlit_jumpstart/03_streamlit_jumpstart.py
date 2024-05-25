# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 0: MACHINE LEARNING & API'S JUMPSTART 
# PART 3: STREAMLIT
# ----

# To run app (put this in Terminal):
#   streamlit run 00_jumpstart/03_streamlit_jumpstart.py


# Import required libraries
import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px

# 1.0 Title and Introduction
st.title("Business Dashboard")
st.write(
    """
    "Welcome to the Business Dashboard. 
    This dashboard provides insights into sales, customers, and products. 
    Upload your data to get started!"
    """
)

# 2.0 Data Input
st.header("Upload Business Data")

uploaded_file = st.file_uploader("Choose a CSV file",
                                 type = "csv",
                                 accept_multiple_files = False)

# 3.0 App Body 
#  What Happens Once Data Is Loaded?

# data = pl.read_csv("00_jumpstart/06_streamlit_jumpstart/data/sales.csv")

if uploaded_file:
    
    data = pl.read_csv(uploaded_file)
    st.write("Preview of the uploaded data:")
    st.write(data.sample(n = 10))

    # * Sales insights
    st.header("Sales Insights")
    if "sales_date" in data.columns and "sales_amount" in data.columns:
        st.write("Sales Over Time")
        fig = px.line(data.to_pandas(), 
                      x = "sales_date", 
                      y = "sales_amount",
                      title = "Sales Over Time")
        st.plotly_chart(fig)
    else:
        st.warning("Please ensure your data has sales date and sales amount columns for sales visualization.")
        

    # * Customer Segmentation by Region
    st.header("Customer Segmentation")
    if "region" in data.columns and "sales_amount" in data.columns:
        st.write("Customer Segmentation:")
        fig = px.pie(
            data.to_pandas(),
            names = "region",
            values = "sales_amount"
        )
        st.plotly_chart(fig)
    else:
        st.warning("Pleasure ensure your data has a `region` column.")

    # * Product Analysis
    st.header("Product Analysis")
    if "product" in data.columns and "sales_amount" in data.columns:
        st.write("Top products by sales:")
        
        top_product_df = (
            data
            .group_by("product")
            .agg(
                pl.col("sales_amount").sum().alias("total_sales_amount")
            )
            .sort("total_sales_amount", descending = True)
        )
        
        fig = px.bar(
            top_product_df.to_pandas(),
            x = "product",
            y = "total_sales_amount",
            title = "Top Products by Sales"
        )
        
        st.plotly_chart(fig)

    else:
        st.warning("Ensure that `product` is in data for sales analysis.")

    # * Feedback Form
    st.header("Feedback")
    feedback = st.text_area("Please provide feedback on the dashboard.")
    if st.button("Submit Feedback"):
        st.success("Feedback Submitted. Thank you!")
    
# 4.0 Footer
st.write("---")
st.write("Business Dashboard by Business Science")


if __name__ == "__main__":
    pass