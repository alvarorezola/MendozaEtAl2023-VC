# importamos paquetes
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import math

def summary_with_stars(model):
    # get the summary table as a DataFrame
    summary_df = model.summary2().tables[1]

    # create a new column for stars
    summary_df['stars'] = ''

    # add stars based on p-values
    summary_df.loc[summary_df['P>|z|'] < 0.001, 'stars'] = '***'
    summary_df.loc[(summary_df['P>|z|'] >= 0.001) & (summary_df['P>|z|'] < 0.01), 'stars'] = '**'
    summary_df.loc[(summary_df['P>|z|'] >= 0.01) & (summary_df['P>|z|'] < 0.05), 'stars'] = '*'

    # return the modified summary table
    return summary_df

# import data
stata_dataset = "/home/alvaro/Desktop/MendozaEtAl2023-VC/Data/CROWD_SUSTAINABILITY_FINAL.dta"
df = pd.read_stata(stata_dataset)
df.drop(index=range(3679,len(df)), inplace=True)

df = df[df["form_c"] ==1]
# 1,853 investment crowdfunding campaings under the Form C exemption from May 2016 until September 2019

df["deadline"] = pd.to_datetime(df["deadline"])
df = df[df["deadline"] <= "2019-10-01"]
# 1,768 investment crowdfunding campaigns

# probit regressions to find suitable independent variables
df = df.dropna(axis = 1)
all_vars = df.columns.tolist()
indep_vars = [var for var in all_vars if var not in ["sustainable",
                                                     "exito",
                                                     "operation_id",
                                                     "cik",
                                                     "id_empresa",
                                                     "success",
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
                                                     "platform_state"]]
indep_vars_str = " + ".join(indep_vars)

formula_treatment = f"sustainable ~ {indep_vars_str}"
formula_outcome = f"exito ~ {indep_vars_str}"

model_treatment = smf.probit(formula=formula_treatment, data=df).fit()
summary_with_stars(smf.probit(formula="sustainable ~ employees + asked + sizemostrecent1", data=df).fit()) 

model_outcome = smf.probit(formula=formula_outcome, data=df).fit()
summary_with_stars(smf.probit(formula="exito ~ logemployees1 + asked1 + sizemostrecent1", data=df).fit())
# logit model para estimar el puntaje de propensión (ps)
model = LogisticRegression()
df = df.dropna(axis = 1) # drop all variables that have empty values
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
            axis = 1) # variables independientes (variables de texto también eliminadas)
y = df["sustainable"] # variable dependiente (grupo de tratamiento)
model.fit(X, y) # ajustar el modelo
pred_prob = model.predict_proba(X) # obtener las probabilidades predichas

df["ps"] = pred_prob[:, 1]

# calculamos el logit del puntuaje de propensión para emparejar
def logit(p):
    logit_value = math.log(p/(1 - p))
    return logit_value

df["ps_logit"] = df.ps.apply(logit)

# definimos funciones de emparejamiento
# Caliper
def caliper_match(df, threshold):
    # ordenar los datos por ps_logit y crear una columna con el índice original
    df_sorted = df.sort_values("ps_logit").reset_index()
    df_sorted["orig_index"] = df_sorted.index
    
    # crear listas vacias para almacenar los indices emparejados y no emparejados
    matched_index = []
    unmatched_index = []
    
    # iterar sobre las filas del dataframe ordenado
    for i in range(len(df_sorted)):
        row = df_sorted.iloc[i]
        if i not in matched_index: # si la fila no está emparejada todavía
            potential_matches = df_sorted[(df_sorted.sustainable != row.sustainable) & (abs(df_sorted.ps_logit - row.ps_logit) <= threshold)]
            
            # encontrar las filas potenciales que tienen un tratamiento diferente y una diferencia de ps_logit menor o igual al umbral
            
            if len(potential_matches) > 0: # si hay al menos una fila potencialmente emparejable
                closest_match_index = potential_matches.iloc[0].orig_index
                # Tomar la priemra fila potencial como la más cercana
                
                matched_index.append(i)
                matched_index.append(closest_match_index)
                # añadir ambos índices a la lista de emparejados
                
            else:
                unmatched_index.append(i)
                # si no hay ninguna fila potencialmente emparejable, añadir el índice a la lista d eno emparejados
    return matched_index, unmatched_index

# Nearest match
def nearest_match(df):
    #ordenar los datos por ps_logit y crear una columna con el indice original
    df_sorted = df.sort_values("ps_logit").reset_index()
    df_sorted["orig_index"] = df_sorted.index
    
    # crear listas vacias para almacenar los índices emparejados y no emparejados
    matched_index = []
    unmatched_index = []
    
    # Iterar sobre las filas del dataframe ordenado
    for i in range(len(df_sorted)):
        row = df_sorted.iloc[i]
        if i not in matched_index: # si la fila no está emparejada todavía
            potential_matches 
            
# emparejar por mas cercano 1-a-1 con remplazo
def nearest_match_with_replacement(df):
    df_sorted = df.sort_values("ps_logit").reset_index()
    df_sorted["orig_index"] = df_sorted.index
    
    matched_index = []
    unmatched_index = []
    
    for i in range(len(df_sorted)):
        row = df_sorted.iloc[i]
        if i not in matched_index:
            potential_matches = df_sorted[(df_sorted.sustainable != row.sustainable)]
            
            if len(potential_matches) > 0:
                closest_match_index = potential_matches.iloc[(potential_matches.ps_logit - row.ps_logit).abs().argsort()[0]].orig_index
                
                matched_index.append(i)
                matched_index.append(closest_match_index)
            
            else:
                unmatched_index.append(i)
    
    return matched_index, unmatched_index

# Aplicar cada una de las funciones de emparejamiento al DF y obtener los indices emparejados y no emparjeados
caliper_matched, caliper_unmatched = caliper_match(df, 0.1) # umbral 0.2
nearest_match, nearest_unmatched = nearest_match(df)
nearest_match_with_replacement_matched, nearest_match_with_replacement_unmatched = nearest_match_with_replacement(df)

# crear el subcojunto del dataframe original con los indices emparejados y calcular el efecto promedio del tratamiento (ATE) para cada variacion
caliper_df = df.iloc[caliper_matched]
nearest_df = df.iloc[nearest_match]
nearest__with_replacement_df = df.iloc[nearest_match_with_replacement_matched]

caliper_ate = caliper_df.groupby("sustainable")["exito"].mean().diff().iloc[-1]
nearest_ate = nearest_df.groupby("sustainable")["exito"].mean().diff().iloc[-1]
nearest_with_replacement_ate = nearest__with_replacement_df.groupby("sustainable")["exito"].mean().diff().iloc[-1]

print(f"ATE por calibre: {caliper_ate:.3f}")
print(f"ATE por más cercano sin remplazo: {nearest_ate:.3f}")
print(f"ATE por más cercano con remplazo: {nearest_with_replacement_ate:.3f}")

# calcular error estandar del ATE

def ate_se(df):
    # obtener el numero de observaciones de cada grupo
    n_t = df[df.sustainable == 1].shape[0]
    n_c = df[df.sustainable == 0].shape[0]
    
    # obtener la varianza de los resultados en cada grupo
    s_t = df[df.sustainable == 1].exito.var()
    s_c = df[df.sustainable == 0].exito.var()
    
    # calcular error estandar del ATE usando la fórmula
    se = np.sqrt((s_t / n_t) + (s_c / n_c))
    
    return se

# calcular el error estandar del ATE para cada variacion
caliper_se = ate_se(caliper_df)
nearest_se = ate_se(nearest_df)
nearest__with_replacement_se = ate_se(nearest__with_replacement_df)

# realizar una preuba de t de dos muestras independientes y obtener los valores p y los intervalso de confianza al 95%
caliper_p, caliper_ci, caliper_df = sms.ttest_ind(caliper_df[caliper_df.sustainable == 1].exito.values,
                                                  caliper_df[caliper_df.sustainable == 0].exito.values,
                                                  usevar="unequal",
                                                  alternative="larger", value=0)

nearest_p, nearest_ci, nearest_df = sms.ttest_ind(nearest_df[nearest_df.sustainable == 1].exito.values,
                                                  nearest_df[nearest_df.sustainable == 0].exito.values,
                                                  usevar="unequal",
                                                  alternative="larger", value=0)

nearest_with_replacement_p, nearest_with_replacement_ci, nearest_with_replacement_df = sms.ttest_ind(nearest__with_replacement_df[nearest__with_replacement_df.sustainable == 1].exito.values,
                                                  nearest__with_replacement_df[nearest__with_replacement_df.sustainable == 0].exito.values,
                                                  usevar="unequal",
                                                  alternative="larger", value=0)
# tabla

data = {"Modelo": ["Calibre", "Más cercano sin remplazo","Más cercano con remplazo"],
        "ATE": [caliper_ate, nearest_ate, nearest_with_replacement_ate],
        "Error estándar": [caliper_se, nearest_se, nearest__with_replacement_se],
        "Valor p": [caliper_p, nearest_p, nearest_with_replacement_p],
        "Intervalo de confianza (95%)": [caliper_ci, nearest_ci, nearest_with_replacement_ci]}

df_results = pd.DataFrame(data)

# nearest-neighbor variance bias-corrected matching method (nn-VBC)
