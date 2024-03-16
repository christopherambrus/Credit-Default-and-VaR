import quandl
import pandas as pd
import nasdaqdatalink
from sklearn.linear_model import LogisticRegression

quandl.ApiConfig.api_key = 'WtSSjYZDFRsDuTXCMRsq'
nasdaqdatalink.ApiConfig.api_key = 'WtSSjYZDFRsDuTXCMRsq'

unemp = quandl.get('FRED/UNRATE')
df_unemp = pd.DataFrame(unemp)
coreUS = nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2005-12-31', paginate=True)
df_coreUS = pd.DataFrame(coreUS)

df_coreUS['calendardate'] = pd.to_datetime(df_coreUS['calendardate'])

df_coreUS['Current Ratio'] = df_coreUS['assetsc'] / df_coreUS['liabilitiesc']
df_coreUS['Debt to Equity Ratio'] = df_coreUS['debt'] / df_coreUS['equity']
df_coreUS['Net Profit Margin'] = df_coreUS['netinc'] / df_coreUS['revenue']
df_coreUS['Return on Equity'] = df_coreUS['netinc'] / df_coreUS['equity']

df_coreUS_annual = df_coreUS.groupby(['ticker', pd.Grouper(key='calendardate', freq='A')]).agg({
    'Current Ratio': 'mean',
    'Debt to Equity Ratio': 'mean',
    'Net Profit Margin': 'mean',
    'Return on Equity': 'mean'
}).reset_index()

print(df_coreUS_annual)
