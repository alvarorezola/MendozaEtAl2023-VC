import pandas as pd
import locale
import numpy as np
from psmpy import psm
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from linearmodels.system import IVSystemGMM
from linearmodels.iv import IVGMM
from linearmodels import PooledOLS, PanelOLS, RandomEffects

# import data
stata_dataset = "/home/alvaro/Desktop/MendozaEtAl2023-VC/Data/CROWD_SUSTAINABILITY_FINAL.dta"
df = pd.read_stata(stata_dataset)
df.drop(index=range(3679,len(df)), inplace=True)

df = df[df["form_c"] ==1]
# 1,853 investment crowdfunding campaings under the Form C exemption from May 2016 until September 2019

df["deadline"] = pd.to_datetime(df["deadline"])
df = df[df["deadline"] <= "2019-10-01"]
# 1,768 investment crowdfunding campaigns

# Table 1: Descriptive statistics
df[["exito",
    "quick75relative",
    "sustainable",
    "totalassetsmostrecent",
    "employees",
    "age",
    "equity",
    "asked",
    "loglagnum_oper_por_platf_y", # Number of offerings per platform
    "lagbranches",                 # Bank branches
    "lagvcfundraising"            # VC fundraising
    ]].describe()
# No encuentro Bank Net Income to Total Assets

df_sust = df[df["sustainable"] == 1]
df_non_sust = df_sust = df[df["sustainable"] == 0]

# calculate t-statistic and p-value for each pair
data = {
    "exito": (df_non_sust["exito"],df_sust["exito"]),
    "quick75relative": (df_non_sust["quick75relative"],df_sust["quick75relative"]),
    "totalassetsmostrecent": (df_non_sust["totalassetsmostrecent"],df_sust["totalassetsmostrecent"]),
    "employees": (df_non_sust["employees"],df_sust["employees"]),
    "age": (df_non_sust["age"],df_sust["age"]),
    "equity": (df_non_sust["equity"],df_sust["equity"]),
    "asked": (df_non_sust["asked"],df_sust["asked"]),
    "loglagnum_oper_por_platf_y": (df_non_sust["loglagnum_oper_por_platf_y"],df_sust["loglagnum_oper_por_platf_y"]),
    "lagbranches": (df_non_sust["lagbranches"],df_sust["lagbranches"]),
    "lagvcfundraising": (df_non_sust["lagvcfundraising"],df_sust["lagvcfundraising"]),
}

results = {}

for key in data:
    group1 = data[key][0]
    group2 = data[key][1]

    # calculate t-statistic    
    t_statistic, p_value = stats.ttest_ind(group1, group2, nan_policy="omit")
    
    # Indicate statistical significance at different levels
    if p_value < 0.01:
        significance = "***"
    elif p_value < 0.05:
        significance = "**"
    elif p_value < 0.10:
        significance = "*"
    else:
        significance = ""
        
    # calculate mean of each variable
    mean_group1 = np.mean(group1)
    mean_group2 = np.mean(group2)

    results[key]={'non sustainable':mean_group1,
                  'sustainable':mean_group2,
                  't-statistic':t_statistic,
                  'p-value':p_value,
                  'significance':significance}
# Display results
df_results = pd.DataFrame(results).T

# Propensity score matching
covariates = ["exito", "quick75relative", "totalassetsmostrecent", "employees", "asked"]
# caliper 1-to-1

# Calculate t-statistic for mean differences between two groups after matching
t_statistic,p_value = stats.ttest_ind(matched_data[matched_data['treated']==1]['outcome'],
                                      matched_data[matched_data['treated']==0]['outcome'])

print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")
# nearest 1-to-1

# nn-VBC


# set the locale to the United States
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
# usd = locale.currency(NUMBER, grouping=True)

# histogram of Form C raised funds
plt.hist(df[(df["raised"] > 10000) & (df["form_c"] == 1)]["raised"], 50000, edgecolor="black", linewidth=1.2)
plt.xlabel("Raised")
plt.ylabel("Density")
plt.title("Histogram")

# univariate ols
formula = "raised ~ logtotalassetsrecent"
results = smf.ols(formula, df).fit()
results.summary()

# F test for another hypothesis
hypothesis = "logtotalassetsrecent = 1"
f_test = results.f_test(hypothesis)

# scatter plot
plt.figure(2)
plt.scatter(df["raised"], df["logtotalassetsrecent"])
plt.ylabel("logtotalassetsrecent")
plt.xlabel("raised")

# multivariate ols
formula = "raised ~ logtotalassetsrecent + sustainable + asked"
results = smf.ols(formula, df).fit()


# adding a dummy variable
# df["dummy-name"] = np.where(df.index == "filter", 1, 0)
# then add the "dummy-name" to the formula definition in the regression

# IV
# formula = 'inflation ~ 1 + dprod + dcredit + dmoney + [rsandp ~ rterm + dspread]'
# mod = IVGMM.from_formula(formula, data, weight_type='unadjusted')
# res1 = mod.fit(cov_type='robust')
# print(res1.summary)

# logit regression
formula = "sustainable ~ logtotalassetsrecent + raised"
mod = smf.logit(formula, df)
res = mod.fit(cov_type="HC1")
res.summary()

# probit regression
formula = "sustainable ~ logtotalassetsrecent + raised"
mod = smf.probit(formula, df)
res = mod.fit(cov_type="HC1")
res.summary()

plt.plot(res.predict())
plt.ylabel("Pr(Sustainable)")
plt.xlabel("Total Assets + Raised")

print(res.get_margeff().summary())