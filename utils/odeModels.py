# ====================================================================================
# ODE models
# ====================================================================================
import numpy as np
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import seaborn as sns
sns.set(style="white")
sys.path.append("./")
from odeModelClass import ODEModel

# ======================== House Keeping Funs ==========================================
def MakeModelFromStr(modelName,**kwargs):
    funList = {"Exponential":Exponential, "Logistic":Logistic, "Gompertz":Gompertz,
               "vonBertalanffy":vonBertalanffy, "GeneralisedLogistic":GeneralisedLogistic,
               "CycleArrestModel_singleStep":CycleArrestModel_singleStep}
    return funList[modelName](**kwargs)

# ======================= Growth Models =======================================
# --------------------- Exponential -----------------------------------
class Exponential(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Exponential"
        self.paramDic = {**self.paramDic,
            'r': 0.03,
            'N0': 10}
        self.stateVars = ['N']

    # The governing equations
    def ModelEqns(self, t, uVec):
        N, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['r'] * N
        dudtVec[1] = 0
        return (dudtVec)


# ---------------------- Logistic -----------------------------------
class Logistic(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Logistic"
        self.paramDic = {**self.paramDic,
            'r': 0.03,
            'K': 100,
            'N0': 10}
        self.stateVars = ['N']

    # The governing equations
    def ModelEqns(self, t, uVec):
        N, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['r'] * (1 - N / self.paramDic['K']) * N
        dudtVec[1] = 0
        return (dudtVec)


# ---------------------- Gompertz -----------------------------------
class Gompertz(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Gompertzian"
        self.paramDic = {**self.paramDic,
            'r': 0.03,
            'K': 100,
            'N0': 10}
        self.stateVars = ['N']

    # The governing equations
    def ModelEqns(self, t, uVec):
        N, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['r'] * np.log(self.paramDic['K'] / N) * N
        dudtVec[1] = 0
        return (dudtVec)


# ----------------- Generalised Logistic -------------------------
class vonBertalanffy(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "vonBertalanffy"
        self.paramDic = {**self.paramDic,
            'r': 0.03,
            'N0': 10}
        self.stateVars = ['N']

    # The governing equations
    def ModelEqns(self, t, uVec):
        N, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['r'] * np.power(N, 2 / 3)
        dudtVec[1] = 0
        return (dudtVec)

# ----------------- Generalised Logistic -------------------------
class GeneralisedLogistic(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Generalised_Logistic"
        self.paramDic = {**self.paramDic,
            'r': 0.03,
            'K': 100,
            'v': 2 / 3,
            'N0': 10}
        self.stateVars = ['N']

    # The governing equations
    def ModelEqns(self, t, uVec):
        N, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['r'] * (1 - np.power(N / self.paramDic['K'], self.paramDic['v'])) * N
        dudtVec[1] = 0
        return (dudtVec)

# ======================= Treatment Models =======================================
# ----------------- Single Step Model -------------------------
class CycleArrestModel_singleStep(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CycleArrestModel_singleStep"
        self.paramDic = {**self.paramDic,
            'r': 0.03,
            'K': 100,
            'v': 2 / 3,
            'd_A':1.,
            'alpha': 0.01,
            'beta': 0.01,
            'P0': 10,
            'A0':0}
        self.stateVars = ['P', 'A']

    # The governing equations
    def ModelEqns(self, t, uVec):
        P, A, D = uVec
        D_hat = D/self.paramDic['DMax']
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['r'] * (1 - np.power((P + A) / self.paramDic['K'], self.paramDic['v'])) * (1-2*self.paramDic['alpha']*D_hat) * P + self.paramDic['beta']*A
        dudtVec[1] = self.paramDic['alpha'] * self.paramDic['r'] * (1 - np.power((P + A) / self.paramDic['K'], self.paramDic['v'])) * D_hat * P - (self.paramDic['d_A']+self.paramDic['beta'])*A
        dudtVec[2] = 0
        return (dudtVec)