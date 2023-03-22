# importamos paquetes
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.stats.api as sm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math

# import data
stata_dataset = "/home/alvaro/Desktop/MendozaEtAl2023-VC/Data/CROWD_SUSTAINABILITY_FINAL.dta"
df = pd.read_stata(stata_dataset)
df.drop(index=range(3679,len(df)), inplace=True)

df = df[df["form_c"] ==1]
# 1,853 investment crowdfunding campaings under the Form C exemption from May 2016 until September 2019

df["deadline"] = pd.to_datetime(df["deadline"])
df = df[df["deadline"] <= "2019-10-01"]
all_vars = df.columns.tolist()

# fixed effects variables
year_dummies = [""]
industry_year_dummies = [""]
state_year_dummies = [""]

# 1,768 investment crowdfunding campaigns

# probit regressions to find suitable independent variables
df = df.dropna(axis = 1)

# Specify and estimate the model
# 1 Run the first stage regression
first_stage_1 = sm.OLS.from_formula('sustainable ~ desastre12meses + employees + asked + sizemostrecent1 + age + equity + '
                                    + " + ".join(year_dummies.columns)
                                    + " + ".join(state_year_dummies)
                                    + " + ".join(industry_year_dummies), df)
first_stage_res = first_stage_1.fit()

# 2 Run the first stage regression
first_stage_2 = sm.OLS.from_formula('sustainable ~ desastre12meses_indemnizaciones + employees + asked + sizemostrecent1 + age + equity + '
                                    + " + ".join(year_dummies.columns)
                                    + " + ".join(state_year_dummies)
                                    + " + ".join(industry_year_dummies), df)

# 2
data_2sls = df.copy()
data_2sls['endogenous_variable_fitted'] = first_stage_res.fittedvalues
second_stage = sm.OLS.from_formula('exito ~ employees + asked + sizemostrecent1 + age + equity + endogenous_variable_fitted', data_2sls)
second_stage_res = second_stage.fit()
    
mod = IV2SLS.from_formula('dependent_variable ~ exogenous_variables + [endogenous_variable ~ instruments]', data)
res = mod.fit()
# 3
mod = IV2SLS.from_formula('dependent_variable ~ exogenous_variables + [endogenous_variable ~ instruments]', data)
res = mod.fit()
# 4
mod = IV2SLS.from_formula('dependent_variable ~ exogenous_variables + [endogenous_variable ~ instruments]', data)
res = mod.fit()
# 5
mod = IV2SLS.from_formula('dependent_variable ~ exogenous_variables + [endogenous_variable ~ instruments]', data)
res = mod.fit()
# 6
mod = IV2SLS.from_formula('dependent_variable ~ exogenous_variables + [endogenous_variable ~ instruments]', data)
res = mod.fit()