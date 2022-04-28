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
               "CycleArrestModel_singleStep":CycleArrestModel_singleStep,
               "MultiStepModel_proliferating":MultiStepModel_proliferating}
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

# ------------ Multi-step model that allows for n rounds of divisions before cell goes into arrest ------------
class MultiStepModel_proliferating(ODEModel):
    def __init__(self, n_steps=1, **kwargs):
        super().__init__(**kwargs)
        self.name = "MultiStepModel_proliferating"
        self.paramDic = {**self.paramDic,
            **dict([['r%d'%x,0.03] for x in range(0, n_steps)]),
            'K': 100,
            'v': 2 / 3,
            **dict([['alpha%d'%x,0.01] for x in range(0, n_steps)]),
            'd_A':0.15,
            'beta': 0.01,
            'P00': 10,
            **dict([['P%d0'%x,0] for x in range(1, n_steps)]),
            'A0':0}
        self.stateVars = [*['P%d' % x for x in range(n_steps)], 'A']
        self.n_pops = n_steps
        self.SetParams()

    # The governing equations
    def ModelEqns(self, t, uVec):
        N = np.sum(uVec[:-1])
        D_hat = uVec[-1]/self.paramDic['DMax']
        dudtVec = np.zeros_like(uVec)
        growth_fraction = (1 - np.power(N / self.paramDic['K'], self.paramDic['v']))
        dudtVec[0] = self.paramDic['r0']*growth_fraction*(1-2*self.paramDic['alpha0']*D_hat)*uVec[0] + self.paramDic['beta']*uVec[1]
        n = 0
        for n in range(1,self.n_pops):
            dudtVec[n] = 2*self.paramDic['alpha%d'%(n-1)]*D_hat*self.paramDic['r%d'%(n-1)]*growth_fraction*uVec[n-1] +\
                         self.paramDic['r%d'%n]*growth_fraction*(1-2*self.paramDic['alpha%d'%n]*D_hat)*uVec[n] -\
                         self.paramDic['beta']*uVec[n] + self.paramDic['beta']*uVec[n+1]
        n += 1
        dudtVec[-2] = self.paramDic['alpha%d'%(n-1)]*D_hat*self.paramDic['r%d'%(n-1)]*growth_fraction*uVec[n-1] -\
                      self.paramDic['beta']*uVec[n] - self.paramDic['d_A']*uVec[n]
        dudtVec[-1] = 0
        return (dudtVec)