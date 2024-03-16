import numpy as np
import pandas as pd

class RiskCalculator:
    def __init__(self, company_data):
        self.company_data = company_data
        self.var = None
        self.cvar = None

    def calculate_returns(self):
        if len(self.company_data['price']) > 1: 
            self.company_data['returns'] = self.company_data['price'].pct_change(fill_method=None)
        else:
            self.company_data['returns'] = pd.Series([np.nan] * len(self.company_data['price']), index=self.company_data.index)

    def calculate_var(self, confidence_level=0.05):
        if 'returns' not in self.company_data or self.company_data['returns'].dropna().empty:
            self.calculate_returns()
        if not self.company_data['returns'].dropna().empty:  
            self.var = np.percentile(self.company_data['returns'].dropna(), 100 * confidence_level)
        else:
            self.var = np.nan  
        return self.var

    def calculate_cvar(self, confidence_level=0.05):
        if 'returns' not in self.company_data or self.company_data['returns'].dropna().empty:
            self.calculate_returns()
        filtered_returns = self.company_data['returns'].dropna()
        if not filtered_returns.empty and self.var is not None:
            self.cvar = filtered_returns[filtered_returns <= self.var].mean()
        else:
            self.cvar = np.nan 
        return self.cvar
