import pandas as pd
import locale
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
    "sizemostrecent",
    "employees",
    "age",
    "equity",
    "asked"]].describe()

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