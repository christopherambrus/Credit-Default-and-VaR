import pandas as pd
import numpy as np
import nasdaqdatalink
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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
        self.data['Interest Coverage Ratio'] = np.where(self.data['intexp'] != 0, self.data['ebit'] / self.data['intexp'], np.nan)
        self.data['Quick Ratio'] = (self.data['assetsc'] - self.data['inventory']) / self.data['liabilitiesc']
        self.data['Cash Ratio'] = self.data['cashneq'] / self.data['liabilitiesc']
        self.data['Operating Margin'] = self.data['opinc'] / self.data['revenue']
        self.data['Asset Turnover Ratio'] = self.data['revenue'] / self.data['assets']

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
        self.year_data['Default'] = self.year_data['ticker'].apply(lambda x: 1 if x in surviving_tickers else 0)

data_2015 = nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2015-12-31', paginate=True)
data_2016 = nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2016-12-31', paginate=True)

companies_2015 = [Company(ticker, data) for ticker, data in data_2015.groupby('ticker')]
companies_2016 = [Company(ticker, data) for ticker, data in data_2016.groupby('ticker')]

for company in companies_2015:
    company.calculate_financial_ratios()
    risk_calculator = RiskCalculator(company.data) 
    var = risk_calculator.calculate_var(0.05)
    cvar = risk_calculator.calculate_cvar(0.05)
    #print(f"{company.ticker} - VaR: {var}, CVaR: {cvar}") #uncomment to see VaR and CVaR's

analyzer_2015 = FinancialAnalyzer(pd.concat([company.data for company in companies_2015]))
analyzer_2016 = FinancialAnalyzer(pd.concat([company.data for company in companies_2016]))

defaulted_tickers = analyzer_2015.compare_tickers(analyzer_2016.year_data)
analyzer_2015.assign_default_status(defaulted_tickers)

df_coreUS_annual = analyzer_2015.year_data.groupby(['ticker', pd.Grouper(key='calendardate', freq='A')]).agg({
    'Current Ratio': 'mean',
    'Debt to Equity Ratio': 'mean',
    'Net Profit Margin': 'mean',
    'Return on Equity': 'mean',
    'Interest Coverage Ratio': 'mean',  
    'Quick Ratio': 'mean',              
    'Cash Ratio': 'mean',               
    'Operating Margin': 'mean',         
    'Asset Turnover Ratio': 'mean',     
    'Default': 'first'  
}).reset_index()


print(df_coreUS_annual)

number_of_defaults = df_coreUS_annual['Default'].sum()

print(f"Number of Defaults: {number_of_defaults}")

df_coreUS_annual.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_cols = df_coreUS_annual.select_dtypes(include=[np.number]).columns

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_coreUS_annual[numeric_cols] = imputer.fit_transform(df_coreUS_annual[numeric_cols])

X = df_coreUS_annual[['Current Ratio', 'Debt to Equity Ratio', 'Net Profit Margin', 'Return on Equity', 
                      'Interest Coverage Ratio', 'Quick Ratio', 'Cash Ratio', 'Operating Margin', 
                      'Asset Turnover Ratio']]
y = df_coreUS_annual['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(StandardScaler(), LogisticRegression())
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Coefficients of the logistic regression model:")
print(pipeline.named_steps['logisticregression'].coef_)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
