import pandas as pd
import numpy as np
import janitor as jn
import polars as pl
import plotly.express as px

def cost_calc_monthly_cost_table(
    email_list_size: int = 1e5,
    email_list_growth_rate: float = 0.035,
    sales_emails_per_month: int = 5,
    unsub_rate_per_sales_email: float = 0.005,
    customer_conversion_rate: float = 0.05,
    average_customer_value: float = 2000,
    n_periods: int = 12
):
    """
    Calculates the monthly cost table based on the given parameters.

    Args:
        email_list_size (int, optional): The initial size of the email list. Defaults to 1e5.
        email_list_growth_rate (float, optional): The monthly growth rate of the email list. Defaults to 0.035.
        sales_emails_per_month (int, optional): The number of sales emails sent per month. Defaults to 5.
        unsub_rate_per_sales_email (float, optional): The unsubscribe rate per sales email. Defaults to 0.005.
        customer_conversion_rate (float, optional): The rate at which customers convert. Defaults to 0.05.
        average_customer_value (float, optional): The average value of a customer. Defaults to 2000.
        n_periods (int, optional): The number of periods to calculate the cost table for. Defaults to 12.

    Returns:
        pandas.DataFrame: The cost table containing the following columns:
            - period: The period number.
            - email_list_size_no_growth: The email list size without growth.
            - lost_customers_no_growth: The number of lost customers without growth.
            - cost_no_growth: The cost without growth.
            - email_list_size_with_growth: The email list size with growth.
            - lost_customers_with_growth: The number of lost customers with growth.
            - cost_with_growth: The cost with growth.
    """
    
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

def cost_total_unsub_cost(cost_table_df: pl.DataFrame):
    """
    Calculate the total cost of email unsubscriptions.

    Args:
        cost_table_df (pl.DataFrame): DataFrame containing the cost table.

    Returns:
        pl.DataFrame: DataFrame with the summary of total unsubscribed cost.
    """
    summary_df = (
        cost_table_df
        .select(
            pl.col("cost_no_growth").sum(),
            pl.col("cost_with_growth").sum()
        )
    )

    return summary_df

def cost_simulate_unsub_costs(
    email_list_monthly_growth_rate : float = [0, 0.35],
    customer_conversion_rate : float = [0.04, 0.05, 0.06],
    **kwargs
):
    """
    Simulates the unsubscribed costs based on the email list monthly growth rate and customer conversion rate.

    Args:
        email_list_monthly_growth_rate (float, optional): A list of email list monthly growth rates. Defaults to [0, 0.35].
        customer_conversion_rate (float, optional): A list of customer conversion rates. Defaults to [0.04, 0.05, 0.06].
        **kwargs: Additional keyword arguments.

    Returns:
        DataFrame: A DataFrame containing the simulation results with unsubscribed costs.
    """
    
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

def cost_plot_simulated_unsub_cost(simulation_results: pl.DataFrame):
    """
    Plot the cost of unsubscription based on simulated results.

    Args:
        simulation_results (pl.DataFrame): DataFrame containing simulation results.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object representing the cost plot.
    """
    
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