# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING
# PART 1: DATA UNDERSTANDING & KPIS
# ----

# GOAL: ----
# - Saw high costs, feedback showed problems
# - Now need to work with departments to collect data and develop project KPIs

# LIBRARIES ----

# Data Analysis:
import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
import plotly.express as px
import hvplot

# New Libraries:
import sweetviz as sv
import sqlalchemy as sql
from skimpy import skim

# Email Lead Scoring: 
import email_lead_scoring as els

# ?els.cost_calc_monthly_unsub_cost_table

els.cost_simulate_unsub_costs(
    email_list_monthly_growth_rate=np.linspace(0, 0.10, 100),
    customer_conversion_rate=np.linspace(0.0, 0.10, 100),
    sales_emails_per_month=12,
    unsub_rate_per_sales_email=0.01,
    email_list_size=1e5
) \
    .pipe(function = els.cost_plot_simulated_unsub_cost)

# 1.0 CONNECTING TO SQLITE DATABASE ----

engine = sql.create_engine('sqlite:///00_database/crm_database.sqlite')

conn = engine.connect()

sql.inspect(engine).get_table_names()

def read_table(table_name, conn):
    """Reads a table from a SQL Database
    Args:
        table_name (string): name of the table
        conn (connection object): connection to the database

    Returns:
        pl.DataFrame: polars dataframe 
    """
    engine = sql.create_engine('sqlite:///00_database/crm_database.sqlite')
    with engine.connect() as conn:
        return pl.read_database(f'SELECT * FROM {table_name}', conn)
    


# 2.0 COLLECT DATA ----

# Products ----

products_df = read_table('Products', conn)

products_df = (
    products_df
    .with_columns(
        pl.col("product_id").cast(pl.Int64)
    )
)

# Subscribers ----

subscribers_df = read_table('Subscribers', conn)

subscribers_df = (
    subscribers_df
    .with_columns(
        pl.col("mailchimp_id").cast(pl.Int64),
        pl.col("member_rating").cast(pl.Int8),
        pl.col("optin_time").cast(pl.Date)
    )
)

# Tags ----

tags_df = read_table('Tags', conn)

tags_df = (
    tags_df
    .with_columns(
        pl.col("mailchimp_id").cast(pl.Int64)
    )
)

# Transactions ----

transactions_df = read_table('Transactions', conn)

transactions_df = (
    transactions_df
    .with_columns(
        pl.col("product_id").cast(pl.Int64),
        pl.col("purchased_at").cast(pl.Date),
    )
)

# Website ----

website_df = read_table('Website', conn)

website_df = (
    website_df
    .with_columns(
        pl.col("date").cast(pl.Date),
        cs.float().cast(pl.Int64)
    )
)

# 3.0 COMBINE & ORGANIZE DATA ----
# - Problem is related to probability of purchase from email list
# - Need to understand what increases probability of purchase
# - Learning Labs could be a key event
# - Website data would be interesting but can't link it to email
# - Products really aren't important to our initial question - just want to know if they made a purchase or not and identify which are likely

# Make Target Feature

emails_made_purchase = transactions_df['user_email'].unique()

subscribers_df = (
    subscribers_df
    .with_columns(
        pl.when(pl.col("user_email").is_in(emails_made_purchase))
        .then(1)
        .otherwise(0)
        .alias("made_purchase")
    )
)

# Who is purchasing?

(
    subscribers_df
    .select(
        pl.col("made_purchase").sum().alias("total_purchases"),
        pl.col("made_purchase").count().alias("total_subscribers"),
    )
    .with_columns(
        (pl.col("total_purchases") * 100 / pl.col("total_subscribers")).alias("purchase_rate")
    )
)

## 4.8% of subscribers have made a purchase


# By Geographic Regions (Countries)

by_geography_df = (
    subscribers_df
    .group_by("country_code")
    .agg(
        pl.sum("made_purchase").alias("total_purchases"),
        pl.count("made_purchase").alias("total_subscribers"),
    )
    .with_columns(
        (pl.col("total_purchases") * 100 / pl.col("total_subscribers")).alias("country_purchase_rate"),
        (pl.col("total_purchases") * 100 / pl.col("total_purchases").sum()).alias("total_purchase_rate")
    )
    .sort("total_purchase_rate", descending = True)
    .with_columns(
        pl.col("total_purchase_rate").cum_sum().alias("cumulative_purchase_rate")
    )
)

# By Tags (Events)

(
    tags_df
    .group_by("tag")
    .agg(
        pl.col("tag").count().alias("n"),
    )
)

user_events_df = (
    tags_df
    .group_by("mailchimp_id")
    .agg(
        pl.count("tag").alias("tag_count"),
    )
)

(
    subscribers_df
    .join(user_events_df, on = "mailchimp_id", how = "left")
    .with_columns(
        pl.col("tag_count").fill_null(0)
    )
)



# 4.0 SWEETVIZ EDA REPORT




# 5.0 DEVELOP KPI'S ----
# - Reduce unnecessary sales emails by 30% while maintaing 99% of sales
# - Segment list - apply sales (hot leads), nuture (cold leads)

# EVALUATE PERFORMANCE -----


    

# WHAT COULD BE MISSED?
# - More information on which tags are most important






