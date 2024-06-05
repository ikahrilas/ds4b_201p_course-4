import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
import sqlalchemy as sql

def db_read_els_data(
    conn_string: str = 'sqlite:///00_database/crm_database.sqlite',
):
    """Reads raw data from the database and combines it into a single dataframe

    Args:
        conn_string (string, required): Connection string to the database. Defaults to 'sqlite:///00_database/crm_database.sqlite'.

    Returns:
        polars DataFrame: Combined raw data
    """
    # connect to engine
    engine = sql.create_engine(conn_string)
    
    # raw data collect
    with engine.connect() as conn:
        
        # subscribers
        subscribers_df = (
            pl.read_database('SELECT * FROM Subscribers', conn)
            .with_columns(
                pl.col("mailchimp_id").cast(pl.Int64),
                pl.col("member_rating").cast(pl.Int8),
                pl.col("optin_time").cast(pl.Date)
            )
        )
        
        # tags
        tags_df = (
            pl.read_database('SELECT * FROM Tags', conn)
            .with_columns(
                pl.col("mailchimp_id").cast(pl.Int64)
            )
        )
        
        # transactions
        transactions_df = (
            pl.read_database('SELECT * FROM Transactions', conn)
            .with_columns(
                pl.col("product_id").cast(pl.Int64),
                pl.col("purchased_at").cast(pl.Date),
            )
        )
        
        # merge tag counts 
        user_events_df = (
            tags_df
            .group_by("mailchimp_id")
            .agg(
                pl.count("tag").alias("tag_count")
            )
        )
        
        subscribers_joined_df = (
            subscribers_df
            .join(user_events_df, on = "mailchimp_id", how = "left")
            .with_columns(
                pl.col("tag_count").fill_null(0)
            )
        )
        
        # merge the target variable  
        
        emails_made_purchase = transactions_df['user_email'].unique()
        
        subscribers_joined_df = (
            subscribers_joined_df
            .with_columns(
                pl.when(pl.col("user_email").is_in(emails_made_purchase))
                .then(1)
                .otherwise(0)
                .alias("made_purchase")
            )
        )
        
        return subscribers_joined_df
    
def db_read_els_table_names(
    conn_string: str = 'sqlite:///00_database/crm_database.sqlite',
):
    """Reads table names from the database

    Args:
        conn_string (string, required): Connection string to the database. Defaults to 'sqlite:///00_database/crm_database.sqlite'.

    Returns:
        list: List of table names
    """
    # connect to engine
    engine = sql.create_engine(conn_string)
    
    conn = engine.connect()

    table_names = sql.inspect(engine).get_table_names()
        
    return table_names

def db_read_raw_els_table(
    table: str = "Products",
    conn_string: str = 'sqlite:///00_database/crm_database.sqlite',
):
    engine = sql.create_engine(conn_string)
    
    with engine.connect() as conn:
        df = pl.read_database(f'SELECT * FROM {table}', conn)
        
    return df