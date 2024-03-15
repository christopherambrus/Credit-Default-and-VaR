import quandl
import pandas as pd
import nasdaqdatalink

quandl.ApiConfig.api_key = 'WtSSjYZDFRsDuTXCMRsq'
nasdaqdatalink.ApiConfig.api_key = 'WtSSjYZDFRsDuTXCMRsq'

unemp = quandl.get('FRED/UNRATE')  # Unemployment Rate from FRED
df_unemp = pd.DataFrame(unemp)

# Fetch the Core US Fundamentals data for all companies for a specific date
# and paginate if the dataset is large
coreUS = nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2011-12-31', paginate=True) 
df_coreUS = pd.DataFrame(coreUS)

# Print the column names to verify the correct column names
print(df_coreUS.columns)

df_coreUS['Current Ratio'] = df_coreUS['assetsc'] / df_coreUS['liabilitiesc']

# Display the DataFrame with the calculated ratios
print(df_coreUS[['ticker', 'Current Ratio']].head())  # Add other ratios as needed

# Display the first few rows of the unemployment DataFrame
print(df_unemp.head())

# Display the full Core US Fundamentals DataFrame
#print(df_coreUS)
