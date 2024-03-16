import pandas as pd
import numpy as np
import nasdaqdatalink
from risk_calculator import RiskCalculator

nasdaqdatalink.ApiConfig.api_key = 'WtSSjYZDFRsDuTXCMRsq'

class Company:
    def __init__(self, ticker, data):
        self.ticker = ticker
        self.data = data
        self.risk_calculator = RiskCalculator(self.data)  # Initialize RiskCalculator with company data

    def calculate_financial_ratios(self):
        self.data['Current Ratio'] = self.data['assetsc'] / self.data['liabilitiesc']
        self.data['Debt to Equity Ratio'] = self.data['debt'] / self.data['equity']
        self.data['Net Profit Margin'] = self.data['netinc'] / self.data['revenue']
        self.data['Return on Equity'] = self.data['netinc'] / self.data['equity']

    def calculate_risk_metrics(self):
        self.risk_calculator.calculate_returns()
        self.var = self.risk_calculator.calculate_var(0.05)
        self.cvar = self.risk_calculator.calculate_cvar(0.05)


class FinancialAnalyzer:
    def __init__(self, year_data):
        self.year_data = year_data

    def compare_tickers(self, other_year_data):
        tickers_current_year = set(self.year_data['ticker'])
        tickers_other_year = set(other_year_data['ticker'])
        return tickers_current_year - tickers_other_year

    def assign_default_status(self, surviving_tickers):
        self.year_data['Default'] = self.year_data['ticker'].apply(lambda x: 0 if x in surviving_tickers else 1)

data_2005 = nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2005-12-31', paginate=True)
data_2023 = nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2023-12-31', paginate=True)

companies_2005 = [Company(ticker, data) for ticker, data in data_2005.groupby('ticker')]
companies_2023 = [Company(ticker, data) for ticker, data in data_2023.groupby('ticker')]

for company in companies_2005:
    company.calculate_financial_ratios()
    risk_calculator = RiskCalculator(company.data) 
    var = risk_calculator.calculate_var(0.05)
    cvar = risk_calculator.calculate_cvar(0.05)
    print(f"{company.ticker} - VaR: {var}, CVaR: {cvar}")

analyzer_2005 = FinancialAnalyzer(pd.concat([company.data for company in companies_2005]))
analyzer_2023 = FinancialAnalyzer(pd.concat([company.data for company in companies_2023]))

defaulted_tickers = analyzer_2005.compare_tickers(analyzer_2023.year_data)
analyzer_2005.assign_default_status(defaulted_tickers)

df_coreUS_annual = analyzer_2005.year_data.groupby(['ticker', pd.Grouper(key='calendardate', freq='A')]).agg({
    'Current Ratio': 'mean',
    'Debt to Equity Ratio': 'mean',
    'Net Profit Margin': 'mean',
    'Return on Equity': 'mean',
    'Default': 'first'  
}).reset_index()

df_coreUS_annual.loc[df_coreUS_annual['ticker'] == 'A', 'Default'] = 0

print(df_coreUS_annual)
