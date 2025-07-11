Inflation Expense Visualiser

This project provides a data-driven model and an interactive application to analyze the impact of inflation on household expenses. By leveraging historical Consumer Price Index (CPI) datasets, it forecasts future inflation and visualizes how a hypothetical monthly expense would change over time.

The application is built using Python and Streamlit, with the core data analysis performed using the pandas and statsmodels libraries. It is designed to deliver clear, actionable insights to help users understand and plan for inflation-driven expense changes.

Key Features
Data Analysis: Utilizes historical CPI data to calculate key metrics, including month-over-month (MoM) and year-over-year (YoY) inflation rates.

Inflation Forecasting: Employs time series forecasting models (specifically, Exponential Smoothing) to project future CPI values.

Expense Projection: Calculates and visualizes the projected change in a user-defined monthly expense based on the inflation forecast.

Interactive Dashboard: A user-friendly Streamlit web application with interactive charts and sliders to explore data and customize forecasts.

Data Export: Functionality to export processed data as CSV files, ready for further analysis in tools like Power BI.

Technology Stack
Python: The core programming language.

pandas: For data loading, preprocessing, and analysis.

Streamlit: To create the interactive web application and user interface.

Plotly Express: For generating dynamic and interactive charts and visualizations.

statsmodels: For time series forecasting using the Exponential Smoothing model.

Power BI (Optional): The project provides a structured data export for building more detailed and advanced dashboards.

How to Run the Application
Follow these steps to set up and run the Inflation Expense Visualiser locally.

1. Prerequisites
Ensure you have Python installed on your system (Python 3.8 or higher is recommended).

2. Clone the Repository
Clone this repository to your local machine:

Bash

git clone https://github.com/your-username/inflation-expense-visualiser.git
cd inflation-expense-visualiser
3. Prepare the Dataset
Make sure you have your CPI dataset (US CPI.csv) in the same directory as the application file. The file should contain at least two columns, Yearmon (in DD-MM-YYYY format) and CPI.

4. Install Dependencies
Install the required Python libraries using pip:

Bash

pip install streamlit pandas plotly statsmodels numpy
5. Run the Application
Start the Streamlit application from your terminal:

Bash

streamlit run your_app_file_name.py
(Replace your_app_file_name.py with the name you saved your Streamlit code as, e.g., inflation_expense.py).

Your browser will automatically open a new tab with the "Inflation Expense Visualiser" dashboard.

Project Structure
inflation-expense-visualiser/
├── US CPI.csv              # Your CPI dataset
├── your_app_file_name.py   # The Streamlit application code
├── README.md               # This file
Power BI Integration
The application includes a feature to export the processed historical and forecasted data into CSV files. You can use these files to build custom dashboards in Power BI, enabling a deeper level of analysis and visualization.

Simply click the "Download" buttons in the app to get the data you need for Power BI.
