import pandas as pd
import numpy as np
import nasdaqdatalink
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
import xgboost as xgb
from risk_calculator import RiskCalculator

nasdaqdatalink.ApiConfig.api_key = 'WtSSjYZDFRsDuTXCMRsq'

class Company:
    def __init__(self, ticker, data):
        self.ticker = ticker
        self.data = data
        self.risk_calculator = RiskCalculator(self.data)  # Initialize RiskCalculator with company data

    def calculate_financial_ratios(self):
        # Profitability Ratios
        self.data['Net Profit Margin'] = self.data['netinc'] / self.data['revenue']
        self.data['Return on Equity'] = self.data['netinc'] / self.data['equity']
        self.data['Return on Assets'] = self.data['netinc'] / self.data['assets']
        self.data['Gross Profit Margin'] = (self.data['revenue'] - self.data['cor']) / self.data['revenue']
        
        # Leverage Ratios
        self.data['Debt to Equity Ratio'] = self.data['debt'] / self.data['equity']
        self.data['Debt Ratio'] = self.data['debt'] / self.data['assets']
        
        # Efficiency Ratios
        self.data['Asset Turnover Ratio'] = self.data['assetturnover']
        self.data['Inventory Turnover Ratio'] = self.data['cor'] / self.data['inventory']
        
        # Liquidity Ratios
        self.data['Current Ratio'] = self.data['assetsc'] / self.data['liabilitiesc']
        self.data['Quick Ratio'] = (self.data['assetsc'] - self.data['inventory']) / self.data['liabilitiesc']
        self.data['Cash Ratio'] = self.data['cashneq'] / self.data['liabilitiesc']
        
        # Market Value Ratios
        self.data['Earnings Per Share'] = self.data['eps']
        self.data['Price to Earnings Ratio'] = self.data['price'] / self.data['eps']
        self.data['Price to Book Ratio'] = self.data['marketcap'] / (self.data['equity'] + self.data['retearn'])
        self.data['Price to Sales Ratio'] = self.data['marketcap'] / self.data['revenue']
        
        # Cash Flow Ratios
        self.data['Operating Cash Flow Ratio'] = self.data['ncfo'] / self.data['liabilities']
        self.data['Cash Flow to Debt Ratio'] = self.data['ncfo'] / self.data['debt']
        
        # Other Ratios
        self.data['Interest Coverage Ratio'] = self.data['ebit'] / self.data['intexp']
        self.data['EBITDA Margin'] = self.data['ebitda'] / self.data['revenue']

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
    'Asset Turnover Ratio': 'mean',     
    'Return on Assets': 'mean',
    'Gross Profit Margin': 'mean',
    'EBITDA Margin': 'mean',
    'Debt Ratio': 'mean',
    'Cash Flow to Debt Ratio': 'mean',
    'Price to Earnings Ratio': 'mean',
    'Price to Book Ratio': 'mean',
    'Price to Sales Ratio': 'mean',
    'Default': 'first'  
}).reset_index()


print(df_coreUS_annual)

number_of_defaults = df_coreUS_annual['Default'].sum()

print(f"Number of Defaults: {number_of_defaults}")

#Logistic regression 1
df_coreUS_annual.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = df_coreUS_annual.select_dtypes(include=[np.number]).columns

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_coreUS_annual[numeric_cols] = imputer.fit_transform(df_coreUS_annual[numeric_cols])

X = df_coreUS_annual[[
    'Current Ratio', 'Debt to Equity Ratio', 'Net Profit Margin', 'Return on Equity', 
    'Interest Coverage Ratio', 'Quick Ratio', 'Cash Ratio',  
    'Asset Turnover Ratio', 'Return on Assets', 'Gross Profit Margin', 'EBITDA Margin', 
    'Debt Ratio', 'Cash Flow to Debt Ratio', 'Price to Earnings Ratio', 
    'Price to Book Ratio', 'Price to Sales Ratio'
]]
y = df_coreUS_annual['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

pipeline = make_pipeline(StandardScaler(), LogisticRegression())
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

print(classification_report(y_test, y_pred))
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

X_train_with_const = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_with_const).fit()

print(logit_model.summary())

#Logistic regression 2
#Reduce variables to maximize ROC score using through backwards selection
X2 = df_coreUS_annual[[
    'Current Ratio', 'Net Profit Margin', 'Quick Ratio',  
    'Asset Turnover Ratio', 'Price to Earnings Ratio', 'Price to Sales Ratio'
]]

X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=47)
pipeline.fit(X2_train, y_train)

y_pred = pipeline.predict(X2_test)
y_pred_proba = pipeline.predict_proba(X2_test)[:, 1] 

print(classification_report(y_test, y_pred))

X2_train_with_const = sm.add_constant(X2_train)
backward_logit_model = sm.Logit(y_train, X2_train_with_const).fit()

print(backward_logit_model.summary())

predictors = backward_logit_model.params.index
X2_test_with_const = sm.add_constant(X2_test)
y_pred_prob_best = backward_logit_model.predict(X2_test_with_const)

fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_pred_prob_best)
roc_auc_best = auc(fpr_best, tpr_best)

print(f"The AUC of the backwards stepwise logistic model is: {roc_auc_best}")

plt.figure()
plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_best)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Best Subset Selection Model')
plt.legend(loc="lower right")
plt.show()

X3 = df_coreUS_annual[[
    'Current Ratio', 'Quick Ratio','Price to Sales Ratio'
]]

#L1 and L2 regularizations
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=47)
pipeline.fit(X2_train, y_train)
scaler = StandardScaler()
X3_train_scaled = scaler.fit_transform(X3_train)
X3_test_scaled = scaler.transform(X3_test)


lasso_cv = LassoCV(cv=5, random_state=47, max_iter=10000, tol=0.01).fit(X3_train_scaled, y_train)
ridge_cv = RidgeCV(cv=5).fit(X3_train_scaled, y_train)

y_pred_prob_lasso = lasso_cv.predict(X3_test_scaled)
y_pred_prob_ridge = ridge_cv.predict(X3_test_scaled)

roc_auc_lasso = roc_auc_score(y_test, y_pred_prob_lasso)
roc_auc_ridge = roc_auc_score(y_test, y_pred_prob_ridge)

print(f"The ROC AUC score for the Lasso model is: {roc_auc_lasso}")
print(f"The ROC AUC score for the Ridge model is: {roc_auc_ridge}")
print(f"The best alpha for Lasso is: {lasso_cv.alpha_}")
print(f"The best alpha for Ridge is: {ridge_cv.alpha_}")

lasso_coefficients = lasso_cv.coef_
ridge_coefficients = ridge_cv.coef_

coefficients_df = pd.DataFrame({
    'Feature': X3.columns,
    'Lasso Coefficients': lasso_coefficients,
    'Ridge Coefficients': ridge_coefficients
})

print(coefficients_df)

#Random Forest
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=47)

rf_classifier.fit(X_train_scaled, y_train)

y_pred = rf_classifier.predict(X_test_scaled)
y_pred_proba = rf_classifier.predict_proba(X_test_scaled)[:, 1]  

roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"The ROC AUC score for the Random Forest model is: {roc_auc}")

feature_importances = rf_classifier.feature_importances_

#print(f"Feature importances: {feature_importances}")

#SVM
svm_classifier = SVC(probability=True, random_state=47)
svm_classifier.fit(X_train_scaled, y_train)
y_pred_proba = svm_classifier.predict_proba(X_test_scaled)[:, 1]  
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"The ROC AUC score for the SVM model is: {roc_auc}")

#K-Nearest Neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5) 
knn_classifier.fit(X_train_scaled, y_train)
y_pred_proba = knn_classifier.predict_proba(X_test_scaled)[:, 1] 
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"The ROC AUC score for the KNN model is: {roc_auc}")

#XGBoost
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', random_state=47)
xgb_classifier.fit(X_train_scaled, y_train)
y_pred_proba = xgb_classifier.predict_proba(X_test_scaled)[:, 1]  
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"The ROC AUC score for the XGBoost model is: {roc_auc}")