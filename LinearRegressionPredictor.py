
import numpy as np #math
import pandas as pd #will be used for basic data management functions
import matplotlib.pyplot as plt #will be used for visualization of data
#from scipy import stats #will be used for certain stats?
import statsmodels.formula.api as smf #more stats?
import os

parent_path = os.path.dirname(os.path.abspath(__file__))


#takes the data and processes it, removing the Rank and Region
data2018 = (pd.read_csv(f"{parent_path}/both.csv", header=0, sep=",")).drop(columns=["Rank","Region","Year"])#, delim_whitespace=True)
headers = data2018.columns.values
#data2018.plot(x="Life_expectancy",y="Score",kind="scatter")


#input for dependent variable
VarD = input(f"Pick a variable we are predicting, the options are {np.array2string(headers)}: ")


#predicts the dependent variable using information information from the line of best fit
def correlator(val, D_Co, Int):
    
    PVARI = val*D_Co + Int
    
    return(PVARI)


#instructions for terminal output
print("If you want to include a variable, type the corresponding number, otherwise write 'no', no other entries will be accepted.")


Weights = []
Scores = []


#finds the correlation for each of the variables in the model and generates the predicted dependent variable using an analysis from each independent variable
for i in headers:  
    VarI = [f"{i}",0]
    model = smf.ols(f'{VarD} ~ {VarI[0]}', data = data2018)
    results = model.fit()
    VarI[1] = results.rsquared_adj
    Coeff = results.params.values
    Intercept = Coeff[0]
    Co1 = Coeff[1]
    v2 = True
    if ((i != VarD)):
        v1=(input(f"{i}: "))
        if (v1 != "no"):
            v1 = (float(v1))
        else: v2 = False
    else: v2 = False

    if (v2 == True):
        Weights.append((float(VarI[1])))
        Scores.append(correlator(v1, Co1,Intercept))
    else: print(f"{i} will not be used as a parameter")




Weights = np.array(Weights)
Scores = np.array(Scores)

#takes a weighted sum and outputs the result
print("Result: " + str((np.dot(Weights,Scores))/(np.sum(Weights))))






