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
df = df.dropna(subset=["desastre12meses"]) # la variable desastre 12 meses tiene dos empty entries
df = df.dropna(axis = 1)
all_vars = df.columns.tolist()

# fixed effects variables
year_dummies = ["yr2c"]
for i in range(3, 19):
    year_dummies.append(f"yr{i}c")
industry_dummies = ["ind2c"]
for i in range(3, 69):
    industry_dummies.append(f"ind{i}c")
state_dummies = ["state2c"]
for i in range(3, 63):
    state_dummies.append(f"state{i}c")
    
# 1,768 investment crowdfunding campaigns

# probit regressions to find suitable independent variables

# Specify and estimate the model
# 1 Run the first stage regression
formula_first_stage_1 = "sustainable ~ desastre12meses + employees + asked + sizemostrecent1 + age + equity"
formula_first_stage_1 += " + " + " + ".join(year_dummies)
formula_first_stage_1 += " + " + " + ".join(state_dummies)
formula_first_stage_1 += " + " + " + ".join(industry_dummies)
first_stage_1 = sm.OLS.from_formula(formula_first_stage_1, df)
first_stage_res = first_stage_1.fit()

independent_varlables_first_stage_1 = ["desastre12meses",
                                       "employees",
                                       "asked",
                                       "sizemostrecent1",
                                       "age",
                                       "equity"]
independent_varlables_first_stage_1.extend(year_dummies)
independent_varlables_first_stage_1.extend(state_dummies)
independent_varlables_first_stage_1.extend(industry_dummies)

y = df["sustainable"]
X = df.drop(["sustainable",
             "exito",
             "filenum",
             "form",
             "datestart",
             "dateend",
             "deadline",
             "jurisdiction",
             "city",
             "state",
             "dateincorporation",
             "cityok",
             "stateok",
             "jurisdictionok",
             "platform_state"],
            axis = 1)

results = sm.Probit(y, df[independent_varlables_first_stage_1]).fit()


formula_first_stage_1 = "desastre12meses", + employees + asked + sizemostrecent1 + age + equity"
formula_first_stage_1 += " + " + " + ".join(year_dummies)
formula_first_stage_1 += " + " + " + ".join(state_dummies)
formula_first_stage_1 += " + " + " + ".join(industry_dummies)

# 2 Run the first stage regression
first_stage_2 = sm.OLS.from_formula('sustainable ~ desastre12meses_indemnizaciones + employees + asked + sizemostrecent1 + age + equity + '
                                    + " + ".join(year_dummies.columns)
                                    + " + ".join(state_dummies)
                                    + " + ".join(industry_dummies), df)
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