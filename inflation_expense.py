import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from datetime import timedelta

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Inflation Expense Visualiser")

# --- Introduction ---
st.title("Inflation Expense Visualiser")
st.markdown("""
Welcome to the **Inflation Expense Visualiser**! This application helps you understand the impact of inflation on your household expenses.
Using historical Consumer Price Index (CPI) data, it calculates inflation rates, forecasts future CPI, and projects how a hypothetical expense
might change over time due to inflation.

**How to use:**
1.  The application automatically loads the `US CPI.csv` file you provided.
2.  Explore the historical CPI trend and inflation rates.
3.  Use the sidebar to set a **forecast horizon** (how many months into the future you want to project).
4.  Enter a **hypothetical monthly expense** in the sidebar (e.g., your average monthly grocery bill).
5.  The app will then show you the forecasted CPI and how your entered expense would change with inflation.
""")

# --- Data Loading and Preprocessing ---
@st.cache_data # Cache the data loading and preprocessing for performance
def load_and_preprocess_data(file_path):
    """
    Loads the CPI data from a CSV file, preprocesses it, and calculates inflation rates.
    """
    try:
        df = pd.read_csv(file_path)
        # Rename columns for clarity and consistency
        df.columns = ['Date', 'CPI_Value']

        # Convert 'Date' column to datetime objects
        # The format is 'DD-MM-YYYY'
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

        # Sort data by date
        df = df.sort_values('Date')

        # Set 'Date' as index for time series operations
        df = df.set_index('Date')

        # Ensure CPI_Value is numeric
        df['CPI_Value'] = pd.to_numeric(df['CPI_Value'], errors='coerce')

        # Drop rows with NaN CPI_Value after conversion
        df.dropna(subset=['CPI_Value'], inplace=True)

        # Calculate Month-over-Month (MoM) CPI change
        df['CPI_MoM_Change'] = df['CPI_Value'].pct_change() * 100

        # Calculate Year-over-Year (YoY) Inflation Rate
        # This calculates the percentage change compared to 12 months prior
        df['YoY_Inflation_Rate'] = df['CPI_Value'].pct_change(periods=12) * 100

        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.stop() # Stop execution if data loading fails

# Path to the uploaded CPI file
cpi_file_path = "US CPI.csv"
cpi_data = load_and_preprocess_data(cpi_file_path)

if cpi_data is not None and not cpi_data.empty:
    st.subheader("1. CPI Data Overview")
    st.write("Here's a glimpse of the processed CPI data:")
    st.dataframe(cpi_data.tail()) # Show the most recent data
    st.write(f"Data available from **{cpi_data.index.min().strftime('%Y-%m')}** to **{cpi_data.index.max().strftime('%Y-%m')}**.")

    # --- Historical CPI Trend ---
    st.subheader("2. Historical CPI Trend")
    fig_cpi_trend = px.line(cpi_data, x=cpi_data.index, y='CPI_Value',
                            title='Historical Consumer Price Index (CPI)',
                            labels={'CPI_Value': 'CPI Value', 'Date': 'Date'})
    fig_cpi_trend.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_cpi_trend, use_container_width=True)

    # --- Historical Inflation Rate ---
    st.subheader("3. Historical Year-over-Year (YoY) Inflation Rate")
    fig_inflation = px.line(cpi_data.dropna(subset=['YoY_Inflation_Rate']), x=cpi_data.dropna(subset=['YoY_Inflation_Rate']).index, y='YoY_Inflation_Rate',
                            title='Year-over-Year Inflation Rate (%)',
                            labels={'YoY_Inflation_Rate': 'Inflation Rate (%)', 'Date': 'Date'},
                            color_discrete_sequence=px.colors.qualitative.Plotly)
    fig_inflation.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0% Inflation", annotation_position="bottom right")
    st.plotly_chart(fig_inflation, use_container_width=True)

    # Display current inflation rate
    latest_inflation = cpi_data['YoY_Inflation_Rate'].iloc[-1]
    st.info(f"**Latest Year-over-Year Inflation Rate:** {latest_inflation:.2f}% (as of {cpi_data.index.max().strftime('%B %Y')})")

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Inflation Forecast & Expense Planning")

    forecast_months = st.sidebar.slider(
        "Select Forecast Horizon (Months)",
        min_value=6,
        max_value=60,
        value=12,
        step=6,
        help="Choose how many months into the future to forecast CPI and expenses."
    )

    initial_expense = st.sidebar.number_input(
        "Enter Your Current Monthly Expense ($)",
        min_value=1.0,
        value=1000.0,
        step=50.0,
        help="Input a hypothetical monthly expense (e.g., your average grocery bill) to see its inflation-adjusted value."
    )

    # --- Forecasting CPI ---
    st.subheader(f"4. CPI Forecast for the Next {forecast_months} Months")

    @st.cache_data
    def forecast_cpi_values(data, months):
        """
        Performs Exponential Smoothing forecasting on CPI data.
        """
        # Use the CPI_Value for forecasting
        series = data['CPI_Value']

        # Determine seasonality. Given monthly data, a 12-month seasonality is typical.
        # If the data length is too short for 12 months, adjust.
        seasonal_periods = 12 if len(series) >= 24 else 1 # At least 2 full cycles for seasonality

        # Fit Exponential Smoothing model
        # Using additive trend and additive seasonality for CPI
        try:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            ).fit()
            forecast = model.forecast(months)
            return forecast
        except Exception as e:
            st.warning(f"Could not fit Exponential Smoothing model (e.g., not enough data for seasonality). Falling back to simpler forecast. Error: {e}")
            # Fallback to a simpler forecast if ExponentialSmoothing fails
            # This could be a simple linear trend or just extending the last known value
            last_cpi = series.iloc[-1]
            forecast_dates = [series.index[-1] + timedelta(days=30 * i) for i in range(1, months + 1)]
            return pd.Series([last_cpi] * months, index=forecast_dates, name='CPI_Value_Forecast')


    cpi_forecast = forecast_cpi_values(cpi_data, forecast_months)

    # Create a DataFrame for plotting historical and forecasted CPI
    forecast_df = pd.DataFrame({
        'CPI_Value': cpi_forecast,
        'Type': 'Forecast'
    })
    historical_df = pd.DataFrame({
        'CPI_Value': cpi_data['CPI_Value'],
        'Type': 'Historical'
    })
    combined_cpi_df = pd.concat([historical_df, forecast_df])

    fig_combined_cpi = px.line(combined_cpi_df, x=combined_cpi_df.index, y='CPI_Value', color='Type',
                               title='Historical and Forecasted CPI',
                               labels={'CPI_Value': 'CPI Value', 'Date': 'Date', 'Type': 'Data Type'},
                               color_discrete_map={'Historical': 'blue', 'Forecast': 'red'})
    st.plotly_chart(fig_combined_cpi, use_container_width=True)

    # --- Impact on Expenses ---
    st.subheader(f"5. Impact of Inflation on Your ${initial_expense:,.2f} Monthly Expense")

    # Calculate future expense based on forecasted CPI
    # Get the last known CPI value from historical data
    last_known_cpi = cpi_data['CPI_Value'].iloc[-1]
    last_known_date = cpi_data.index.max()

    # Create a DataFrame for expense projection
    expense_projection = pd.DataFrame(index=cpi_forecast.index)
    expense_projection['Forecasted_CPI'] = cpi_forecast
    expense_projection['Projected_Expense'] = (expense_projection['Forecasted_CPI'] / last_known_cpi) * initial_expense

    # Add the initial expense for comparison
    initial_expense_row = pd.DataFrame({
        'Forecasted_CPI': last_known_cpi,
        'Projected_Expense': initial_expense
    }, index=[last_known_date])

    full_expense_projection = pd.concat([initial_expense_row, expense_projection])

    st.write(f"Your initial monthly expense of **${initial_expense:,.2f}** would change as follows:")

    # Display expense projection table
    st.dataframe(full_expense_projection[['Projected_Expense']].rename(columns={'Projected_Expense': 'Projected Monthly Expense'}).style.format("${:,.2f}"))

    # Plot expense projection
    fig_expense_impact = px.line(full_expense_projection, x=full_expense_projection.index, y='Projected_Expense',
                                 title=f'Projected Monthly Expense (${initial_expense:,.2f} initial) with Inflation',
                                 labels={'Projected_Expense': 'Projected Monthly Expense ($)', 'Date': 'Date'},
                                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig_expense_impact.add_hline(y=initial_expense, line_dash="dash", line_color="gray",
                                 annotation_text=f"Initial Expense: ${initial_expense:,.2f}", annotation_position="top left")
    st.plotly_chart(fig_expense_impact, use_container_width=True)

    # --- Actionable Insights ---
    st.subheader("6. Actionable Insights")
    if not cpi_forecast.empty:
        projected_expense_end = expense_projection['Projected_Expense'].iloc[-1]
        percent_increase = ((projected_expense_end - initial_expense) / initial_expense) * 100
        st.write(f"- Based on the forecast, your **${initial_expense:,.2f}** monthly expense could increase to approximately **${projected_expense_end:,.2f}** in **{forecast_months} months**. This represents a **{percent_increase:.2f}%** increase.")
        st.write("- Understanding these trends can help you adjust your budget and financial planning to account for future price changes.")
        st.write("- Consider reviewing your spending habits and exploring ways to save or invest to mitigate the impact of inflation.")

    # --- Power BI Integration Guidance ---
    st.subheader("7. Power BI Integration (Guidance)")
    st.markdown("""
    While this Streamlit app provides interactive visualizations, you can also leverage the processed data for more advanced dashboards in Power BI.

    **Steps to integrate with Power BI:**
    1.  **Export Processed Data:** You can export the `cpi_data` DataFrame (which includes historical CPI, MoM change, and YoY inflation) and the `full_expense_projection` DataFrame (which includes forecasted CPI and projected expenses) to CSV files.
    2.  **Import into Power BI:** In Power BI Desktop, use "Get Data" -> "Text/CSV" to import these exported files.
    3.  **Build Dashboards:** Use Power BI's powerful visualization tools to create custom charts, tables, and slicers based on the imported data. You can combine the historical and forecasted data to create comprehensive inflation dashboards.
    4.  **Relationships:** If you export multiple related tables (e.g., historical CPI and forecasted expenses), ensure you establish proper relationships between them in Power BI based on common date columns.
    """)

    # --- Export Data for Power BI ---
    st.subheader("8. Export Data for Power BI")
    st.markdown("Click the buttons below to download the processed dataframes as CSV files, ready for import into Power BI.")

    # Convert DataFrame to CSV string for download
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=True).encode('utf-8')

    csv_cpi_data = convert_df_to_csv(cpi_data)
    csv_projected_expenses = convert_df_to_csv(full_expense_projection)

    st.download_button(
        label="Download Processed CPI Data as CSV",
        data=csv_cpi_data,
        file_name='processed_cpi_data.csv',
        mime='text/csv',
        help="Downloads historical CPI, MoM change, and YoY inflation data."
    )

    st.download_button(
        label="Download Projected Expenses Data as CSV",
        data=csv_projected_expenses,
        file_name='projected_expenses.csv',
        mime='text/csv',
        help="Downloads forecasted CPI and projected expense data."
    )

else:
    st.error("CPI data could not be loaded or is empty. Please ensure 'US CPI.csv' is correctly formatted with 'Yearmon' and 'CPI' columns.")

