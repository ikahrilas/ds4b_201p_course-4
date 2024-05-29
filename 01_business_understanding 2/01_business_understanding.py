# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING
# ----

# LIBRARIES ----
import pandas as pd
import numpy as np
import janitor as jn
import polars as pl
import plotly.express as px
import plotnine 

# BUSINESS SCIENCE PROBLEM FRAMEWORK ----

# View Business as a Machine ----


# Business Units: 
#   - Marketing Department
#   - Responsible for sales emails  
# Project Objectives:
#   - Target Subscribers Likely To Purchase
#   - Nurture Subscribers to take actions that are known to increase probability of purchase
# Define Machine:
#   - Marketing sends out email blasts to everyone
#   - Generates Sales
#   - Also ticks some members off
#   - Members unsubscribe, this reduces email growth and profitability
# Collect Outcomes:
#   - Revenue has slowed, Email growth has slowed


# Understand the Drivers ----

#   - Key Insights:
#     - Company has Large Email List: 100,000 
#     - Email list is growing at 6,000/month less 3500 unsub for total of 2500
#     - High unsubscribe rates: 500 people per sales email
#   - Revenue:
#     - Company sales cycle is generating about $250,000 per month
#     - Average Customer Lifetime Value: Estimate $2000/customer
#   - Costs: 
#     - Marketing sends 5 Sales Emails Per Month
#     - 5% of lost customers likely to convert if nutured



# COLLECT OUTCOMES ----
email_list = 100_000
unsub_count_per_sales_email_1 = 500
unsub_rate_1 = unsub_count_per_sales_email_1 / email_list

sales_emails_per_month_1 = 5
conversion_rate_1 = 0.05

lost_customers = email_list * \
    unsub_rate_1 * \
    sales_emails_per_month_1 * \
    conversion_rate_1
    
average_customer_value_1 = 2000

lost_revenue_per_month_1 = lost_customers * average_customer_value_1


# No-growth scenario $3M cost
cost_no_growth_1 = lost_revenue_per_month_1 * 12
print(f"Cost of No Growth: ${cost_no_growth_1:,.0f}")

# 2.5% growth scenario: 
#   amount = principle * ((1+rate)**time)
growth_rate = 3500 / 100_000

100_000 * ((1 + growth_rate) ** 0)

100_000 * ((1 + growth_rate) ** 1)

## loop over 12 months
for month in range(1, 13):
    growth = 100_000 * ((1 + growth_rate) ** month)
    print(f"Month {month}: ${growth:,.0f}")


## cost table

### period

time = 12

period_series = pl.Series("period", np.arange(0, 12))

len(period_series)

cost_table_df = period_series.to_frame()

### email size - no growth

cost_table_df = (
    cost_table_df
    .with_columns(
        pl.lit(email_list).alias("email_list_size_no_growth")
    )
)

### lost customers - no growth

cost_table_df = (
    cost_table_df
    .with_columns(
        (pl.col("email_list_size_no_growth") * 
         unsub_rate_1 * 
         sales_emails_per_month_1 * 
         conversion_rate_1)
        .alias("lost_customers_no_growth")
    )
    .with_columns(
        (pl.col("lost_customers_no_growth") * 
         conversion_rate_1 * 
         average_customer_value_1)
        .alias("lost_revenue_no_growth")
    
    )
)

print(cost_table_df)

## cost no growth

cost_table_df = (
    cost_table_df
    .with_columns(
        (pl.col("lost_customers_no_growth") *
        conversion_rate_1 * 
        average_customer_value_1)
        .alias("cost_no_growth")
    )
)

## email size with growth

cost_table_df = (
    cost_table_df
    .with_columns(
        (pl.col("email_list_size_no_growth") *
        (1 + growth_rate) ** pl.col("period"))
        .alias("email_list_size_with_growth")
    )
)

px.line(
    cost_table_df.to_pandas(),
    x = "period",
    y = ["email_list_size_no_growth", "email_list_size_with_growth"]
) \
    .add_hline(y = 0)
    
# lost customers with growth

cost_table_df = (
    cost_table_df
    .with_columns(
        (pl.col("email_list_size_with_growth") *
         unsub_rate_1 * 
         sales_emails_per_month_1 * 
         conversion_rate_1)
        .alias("lost_customers_with_growth")
    )
)

cost_table_df

# cost with growth

cost_table_df = (
    cost_table_df
    .with_columns(
        (pl.col("lost_customers_with_growth") *
        conversion_rate_1 * 
        average_customer_value_1)
        .alias("cost_with_growth")
    )
)

### hvplot
(
    cost_table_df
    .plot.line(
        x = "period",
        y = ["lost_revenue_no_growth", "cost_with_growth"],
        title = "Cost Analysis"
    )
)

### plotly express
px.line(
    cost_table_df.to_pandas(),
    x = "period",
    y = ["cost_no_growth", "cost_with_growth"]
) \
    .add_hline(y = 0)
    

# If reduce unsubscribe rate by 30%

(
    cost_table_df
    .select(
        pl.col("cost_no_growth").sum(),
        pl.col("cost_with_growth").sum()
    )
    .with_columns(
        ((pl.col("cost_with_growth") - pl.col("cost_no_growth")) / pl.col("cost_no_growth")).alias("cost_increase_percent")
    )
)

# COST CALCULATION FUNCTIONS ----

# Function: Calculate Monthly Unsubscriber Cost Table ----

def cost_calc_monthly_cost_table(
    email_list_size: int = 1e5,
    email_list_growth_rate: float = 0.035,
    sales_emails_per_month: int = 5,
    unsub_rate_per_sales_email: float = 0.005,
    customer_conversion_rate: float = 0.05,
    average_customer_value: float = 2000,
    n_periods: int = 12
):
    
    period_series = pl.Series("period", np.arange(0, n_periods))
    
    cost_table_df = period_series.to_frame()
    
    cost_table_df = (
        cost_table_df
        # Email Size - No Growth
        .with_columns(
            pl.lit(email_list_size).alias("email_list_size_no_growth")
         )
        # Lost Customers - No Growth
        .with_columns(
            (pl.col("email_list_size_no_growth") * 
                unsub_rate_per_sales_email * 
                sales_emails_per_month * 
                customer_conversion_rate)
            .alias("lost_customers_no_growth")
        )
        # Cost - No Growth
        .with_columns(
            (pl.col("lost_customers_no_growth") *
                customer_conversion_rate * 
                average_customer_value)
            .alias("cost_no_growth")
        )
        # Email Size - With Growth
        .with_columns(
            (pl.col("email_list_size_no_growth") *
                (1 + email_list_growth_rate) ** pl.col("period"))
            .alias("email_list_size_with_growth")
        )
        # Lost Customers - With Growth
        .with_columns(
            (pl.col("email_list_size_with_growth") *
                unsub_rate_per_sales_email * 
                sales_emails_per_month * 
                customer_conversion_rate)
            .alias("lost_customers_with_growth")
        )
        # Cost - With Growth
        .with_columns(
            (pl.col("lost_customers_with_growth") *
                customer_conversion_rate * 
                average_customer_value)
            .alias("cost_with_growth")
        )
    )
    
    return cost_table_df

cost_calc_monthly_cost_table(
    email_list_size = 200_000,
    email_list_growth_rate = 0.10,
    sales_emails_per_month = 12,
    unsub_rate_per_sales_email = 0.01,
    customer_conversion_rate = 0.07,
    average_customer_value = 4000,
    n_periods = 12
)

# Function: Sumarize Cost ----

(
    cost_table_df
    .select(
        pl.col("cost_no_growth").sum(),
        pl.col("cost_with_growth").sum()
    )
)

def cost_total_unsub_cost(cost_table_df: pl.DataFrame):
    summary_df = (
     cost_table_df
        .select(
            pl.col("cost_no_growth").sum(),
            pl.col("cost_with_growth").sum()
        )
    )

    return summary_df 

cost_total_unsub_cost(
    cost_calc_monthly_cost_table()   
)

# ARE OBJECTIVES BEING MET?
# - We can see a large cost due to unsubscription
# - However, some attributes may vary causing costs to change


# SYNTHESIZE OUTCOMES (COST SIMULATION) ----
# - Make a cartesian product of inputs that can vary
# - Use list comprehension to perform simulation
# - Visualize results

# Cartesian Product (Expand Grid)

data_dict = {
    "email_list_monthly_growth_rate": np.linspace(0, 0.05, num = 10),
    "customer_conversion_rate": np.linspace(0.04, 0.06, num = 3),
}

parameter_grid = (
    pl.from_pandas(jn.expand_grid(others = data_dict))
    .rename(
        {
            "('email_list_monthly_growth_rate', 0)": "email_list_monthly_growth_rate",
            "('customer_conversion_rate', 0)": "customer_conversion_rate"
        }
    )
)

# list_comprehension

def temporary_function(x, y):
    
    cost_table_df = cost_calc_monthly_cost_table(
        email_list_growth_rate = x,
        customer_conversion_rate = y
    )
    
    summary_df = cost_total_unsub_cost(cost_table_df)
    
    return summary_df

temporary_function(.10, .10)

summary_list = [temporary_function(x, y) for x, y 
                in zip(parameter_grid["email_list_monthly_growth_rate"], 
                       parameter_grid["customer_conversion_rate"])]

# concatenate summary list
simulation_results_df = pl.concat(summary_list)

print(simulation_results_df)

# Function

def cost_simulate_unsub_costs(
    email_list_monthly_growth_rate : float = [0, 0.35],
    customer_conversion_rate : float = [0.04, 0.05, 0.06],
    **kwargs
):
    
    data_dict = {
        "email_list_monthly_growth_rate": email_list_monthly_growth_rate,
        "customer_conversion_rate": customer_conversion_rate,
    }

    parameter_grid = (
        pl.from_pandas(jn.expand_grid(others = data_dict))
        .rename(
            {
                "('email_list_monthly_growth_rate', 0)": "email_list_monthly_growth_rate",
                "('customer_conversion_rate', 0)": "customer_conversion_rate"
            }
        )
    )
    
    # temporary function
    def temporary_function(x, y):
    
        cost_table_df = cost_calc_monthly_cost_table(
            email_list_growth_rate = x,
            customer_conversion_rate = y,
            **kwargs
        )

        summary_df = cost_total_unsub_cost(cost_table_df)

        return summary_df
    
    # list comprehension
    summary_list = [temporary_function(x, y) for x, y 
                    in zip(parameter_grid["email_list_monthly_growth_rate"], 
                           parameter_grid["customer_conversion_rate"])]
    
    # concatenate summary list
    simulation_results_df = pl.concat(summary_list).hstack(parameter_grid)
    
    return simulation_results_df

cost_simulate_unsub_costs(
    email_list_monthly_growth_rate = [0.01, 0.02],
    customer_conversion_rate = [0.04, 0.06],
    email_list_size = 150_000
)


# VISUALIZE COSTS

simulations_results_wide_df = cost_simulate_unsub_costs(
    email_list_monthly_growth_rate = [0.01, 0.02],
    customer_conversion_rate = [0.04, 0.06],
    email_list_size = 100_000
) \
    .drop("cost_no_growth") \
    .pivot(
        index = "email_list_monthly_growth_rate",
        columns = "customer_conversion_rate",
        values = "cost_with_growth"
    )
    
px.imshow(
    simulations_results_wide_df.to_pandas().set_index("email_list_monthly_growth_rate"),
    origin = "lower",
    aspect = 'auto',
    title = "Lead Cost Simulation",
    labels = dict(x = "Customer Conversion Rate", 
                  y = "Email List Monthly Growth Rate",
                  color = "Cost of Unsubscription")
)

# Function: Plot Simulated Unsubscriber Costs

def cost_plot_simulated_unsub_cost(simulation_results: pl.DataFrame):
    
    simulations_results_wide_df = simulation_results \
    .drop("cost_no_growth") \
    .pivot(
        index = "email_list_monthly_growth_rate",
        columns = "customer_conversion_rate",
        values = "cost_with_growth"
    )
    
    plot = px.imshow(
        simulations_results_wide_df.to_pandas().set_index("email_list_monthly_growth_rate"),
        origin = "lower",
        aspect = 'auto',
        title = "Lead Cost Simulation",
        labels = dict(x = "Customer Conversion Rate", 
                      y = "Email List Monthly Growth Rate",
                      color = "Cost of Unsubscription")
    )
    
    return plot

cost_plot_simulated_unsub_cost(cost_simulate_unsub_costs(
    email_list_monthly_growth_rate = np.linspace(0, 0.1, num = 10),
    customer_conversion_rate = np.linspace(0.04, 0.1, num = 5),
    email_list_size = 100_000
))


# ARE OBJECTIVES BEING MET?
# - Even with simulation, we see high costs
# - What if we could reduce by 30% through better targeting?



# - What if we could reduce unsubscribe rate from 0.5% to 0.17% (marketing average)?
# - Source: https://www.campaignmonitor.com/resources/knowledge-base/what-is-a-good-unsubscribe-rate/



# HYPOTHESIZE DRIVERS

# - What causes a customer to convert of drop off?
# - If we know what makes them likely to convert, we can target the ones that are unlikely to nurture them (instead of sending sales emails)
# - Meet with Marketing Team
# - Notice increases in sales after webinars (called Learning Labs)
# - Next: Begin Data Collection and Understanding



