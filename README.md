Portfolio & Risk Analysis Dashboard (I2F)
This project is an interactive application developed using Streamlit. 

It enables users to manage a portfolio of financial assets in real time and compare its performance against the global market (MSCI World).

This tool automates the calculation of key risk indicators used in asset management.

What the application does: 
Value Tracking: Automatic updating of share prices via the Yahoo Finance API.

Performance Calculation: Inclusion of unrealised capital gains, realised sales and dividends.

Professional Risk Analysis:

Sharpe Ratio: To measure return per unit of risk.

Jensen’s Beta & Alpha: To assess market sensitivity and actual outperformance.

Maximum Drawdown: To identify the risk of the maximum historical loss.

Visualisation: 5-year graphical comparison between the portfolio and its benchmark (IWDA.AS).

How to test the tool:
Follow these 3 steps:

1. Download the project
Click the green ‘Code’ button at the top right of this GitHub page.

Select ‘Download ZIP’.

Once the download is complete, extract (unzip) the folder to your desktop.

2. Install the necessary tools (once only)
Open your terminal (or command prompt) and type the following command to install the professional libraries used:

"pip install streamlit yfinance pandas numpy matplotlib"
3. Launch the Dashboard
In your terminal, navigate to the project folder and launch the application with:

"streamlit run"
Note: Replace filename.py with the exact name of your script (e.g. app.py).

The interface will open immediately in your web browser.
