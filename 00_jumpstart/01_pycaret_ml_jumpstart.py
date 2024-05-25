# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 0: MACHINE LEARNING & API'S JUMPSTART 
# PART 1: PYCARET
# ----

# GOAL: Make an introductory ML lead scoring model
#  that can be used in an API

# LIBRARIES

import os
import pandas as pd
import polars as pl
import sqlalchemy as sql
import pycaret.classification as clf


# 1.0 READ DATA ----

# Connect to SQL Database ----
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
    conn = engine.connect()
    return pl.read_database(f'SELECT * FROM {table_name}', conn)
    conn.close()


# * Subscribers ---
subscribers_df = read_table('Subscribers', conn)
subscribers_df.glimpse()
# * Transactions
transactions_df = read_table('Transactions', conn)
transactions_df.glimpse()

# 2.0 SIMPLIFIED DATA PREP ----

## join subscribers and transactions
df = (
    subscribers_df
    .join(
        transactions_df, 
        on = 'user_email', 
        how = 'inner'
    )
)

emails_made_purchase = transactions_df['user_email'].unique()

subscribers_joined_df = (
    subscribers_df
    .with_columns(
        pl.when(pl.col("user_email").is_in(emails_made_purchase))
        .then(1)
        .otherwise(0)
        .alias("made_purchase")
    )
)

subscribers_joined_df.glimpse()

# 3.0 QUICKSTART MACHINE LEARNING WITH PYCARET ----

# * Subset the data ----
df = (
    subscribers_joined_df
    .select(
        ["member_rating",
         "country_code",
         "made_purchase"]
    )
)
df.glimpse()
# * Setup the Classifier ----
clf_1 = clf.setup(
    data = df.to_pandas(),
    target = 'made_purchase',
    train_size = 0.8,
    session_id = 123
)

clf_1.pipeline

# * Make A Machine Learning Model ----
xgb_model = clf_1.create_model('xgboost')

xgb_model_tuned = clf_1.tune_model(xgb_model, n_iter = 100, optimize = 'AUC')

# get help page on create_model() method in pycaret
help(clf_1.create_model)

# * Finalize the model ----
xgb_model_finalized = clf_1.finalize_model(xgb_model)

# * Predict -----
new_df = pl.DataFrame(
    {
        "member_rating": [4],
        "country_code": ['us']
    }
)

clf.predict_model(xgb_model, 
                  new_df.to_pandas(),
                  raw_score = True)

# * Save the Model ----
os.mkdir('00_jumpstart 2/models')

clf.save_model(xgb_model_finalized, '00_jumpstart 2/models/xgb_model_finalized')

# * Load the model -----
clf.load_model('00_jumpstart 2/models/xgb_model_finalized')


# CONCLUSIONS:
# * Insane that we did all of this in 90 lines of code
# * And the model was better than random guessing...
# * But, there are questions that come to mind...

# KEY QUESTIONS:
# * SHOULD WE EVEN TAKE ON THIS PROJECT? (COST/BENEFIT)
# * MACHINE LEARNING MODEL - IS IT GOOD?
# * WHAT CAN WE DO TO IMPROVE THE MODEL?
# * WHAT ARE THE KEY FEATURES IN THE MODEL?
# * CAN WE EXPLAIN WHY CUSTOMERS ARE BUYING / NOT BUYING?
# * CAN THE COMPANY MAKE A RETURN ON INVESTMENT FROM THIS MODEL?


