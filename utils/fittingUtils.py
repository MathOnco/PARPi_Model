# ====================================================================================
# Various functions that I found useful in this project
# ====================================================================================
import re
import numpy as np
import pandas as pd
import sys
import os
import pickle
import scipy
from math import ceil
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sys.path.append("./")
import myUtils as utils
from odeModels import MakeModelFromStr

def residual(params, x, data, model, feature, solver_kws={}):
    model.SetParams(**params.valuesdict())
    converged = False
    max_step = solver_kws.get('max_step',np.inf)
    currSolver_kws = solver_kws.copy()
    while not converged:
        model.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(data), **currSolver_kws)
        converged = model.successB
        max_step = 0.75*max_step if max_step < np.inf else 100
        currSolver_kws['max_step'] = max_step
    # Interpolate to the data time grid
    t_eval = data.Time
    f = scipy.interpolate.interp1d(model.resultsDf.Time,model.resultsDf.TumourSize,fill_value="extrapolate")
    modelPrediction = f(t_eval)
    return (data[feature]-modelPrediction)

def residual_multipleTxConditions(params, x, data, model, feature, solver_kws={}):
    tmpList = []
    for drugConcentration in data.DrugConcentration.unique():
        currData = data[data.DrugConcentration==drugConcentration]
        # Set initial conditions
        if 'N0' in params.valuesdict().keys():
            if params['N0'].vary==False: params['N0'].value = currData.Confluence.iloc[0]
        else:
            if params['P0'].vary==False: params['P0'].value = currData.Confluence.iloc[0]
        tmpList.append(residual(params, x, currData, model, feature, solver_kws={}))
    return np.concatenate(tmpList)

def PerturbParams(params):
    params = params.copy()
    for p in params.keys():
        currParam = params[p]
        if currParam.vary:
            params[p].value = np.random.uniform(low=currParam.min, high=currParam.max)
    return params

def ComputeRSquared(fit,dataDf,feature="Confluence"):
    tss = np.sum(np.square(dataDf[feature]-dataDf[feature].mean()))
    rss = np.sum(np.square(fit.residual))
    return 1-rss/tss

def PlotData(dataDf, feature='Confluence', plotDrugConcentration=True, titleStr="",
             xlim=None, ylim=None, y2lim=100, decorateX=True, decorateY=True, decorateY2=False, markersize=10,
             ax=None, figsize=(10, 8), outName=None, color="black", **kwargs):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Plot the data
    if not np.all(np.isnan(dataDf[feature])):
        ax.plot(dataDf.Time, dataDf[feature],
                linestyle="None", marker="x", markersize=markersize,
                color=color, markeredgewidth=2)

    # Plot the drug concentration
    if plotDrugConcentration:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        drugConcentrationVec = utils.TreatmentListToTS(treatmentList=utils.ExtractTreatmentFromDf(dataDf),
                                                       tVec=dataDf['Time'])
        ax2.fill_between(dataDf['Time'], 0, drugConcentrationVec, color="#8f59e0", step="post", alpha=0.2)
        ax2.set_ylim([0, y2lim])
        ax2.tick_params(labelsize=28)
        if not decorateY2:
            ax2.set_yticklabels("")

    # Format the plot
    if xlim is not None: ax.set_xlim(0,xlim)
    if ylim is not None: ax.set_ylim(0, ylim)
    ax.set_xlabel("")
    ax.set_ylabel("")
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
    ax.set_title(titleStr)
    ax.tick_params(labelsize=28)
    # ax.legend().remove()
    if not decorateX:
        ax.set_xticklabels("")
    if not decorateY:
        ax.set_yticklabels("")
    plt.tight_layout()
    if outName is not None: plt.savefig(outName)

def PlotFit(fitObj, dataDf, linewidth=5, linewidthA=5, titleStr="", legend=True, outName=None, ax=None, solver_kws={}, **kwargs):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    myModel = MakeModelFromStr(fitObj.modelName)
    myModel.SetParams(**fitObj.params.valuesdict())
    myModel.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(dataDf),max_step=1,**solver_kws)
    myModel.Plot(ymin=0, title=titleStr, linewidth=linewidth, linewidthA=linewidthA, ax=ax, plotLegendB=legend, **kwargs)
    PlotData(dataDf, plotDrugConcentration=False, ax=ax, **kwargs)
    if outName is not None: plt.savefig(outName); plt.close()

def LoadFit(modelName, fitId=0, fitDir="./", **kwargs):
    fitObj = pickle.load(open(os.path.join(fitDir, "fitObj_fit_%d.p"%(fitId)), "rb"))
    myModel = MakeModelFromStr(modelName, **kwargs)
    myModel.SetParams(**fitObj.params.valuesdict())
    return fitObj, myModel

def GenerateFitSummaryDf(fitDir="./fits", identifierName=None, identifierId=1):
    fitIdList = [int(re.findall(r'\d+', x)[0]) for x in os.listdir(fitDir) if
                 x.split("_")[0] == "fitObj"]
    identifierDic = {} if identifierName is None else {identifierName: identifierId}
    tmpDicList = []
    for fitId in fitIdList:
        fitObj = pickle.load(open(os.path.join(fitDir, "fitObj_fit_%d.p"%(fitId)), "rb"))
        tmpDicList.append({**identifierDic, "FitId": fitObj.fitId, "ModelName":fitObj.modelName,
                           "AIC": fitObj.aic, "BIC": fitObj.bic, "RSquared": fitObj.rSq,
                           **fitObj.params.valuesdict(),
                           **dict([(x+"_se",fitObj.params[x].stderr) for x in fitObj.params.keys()])})
    return pd.DataFrame(tmpDicList)

def PlotParameterDistribution_points(summaryDf, exampleFit, x="PatientId", showAll=False, nCols=5, figsize=(12, 4), ax=None):
    paramNamesList = list(exampleFit.params.keys()) if showAll else exampleFit.var_names
    nParams = len(paramNamesList)

    if ax is None: fig, axList = plt.subplots(nParams // nCols + 1, nCols, figsize=figsize)
    nPatients = summaryDf.shape[0]
    for i, param in enumerate(paramNamesList):
        currAx = axList.flatten()[i]
        sns.stripplot(x=x, y=param, data=summaryDf, ax=currAx)
        currAx.hlines(xmin=-1, xmax=nPatients+1, y=exampleFit.params[param].min, linestyles='--')
        currAx.hlines(xmin=-1, xmax=nPatients+1, y=exampleFit.params[param].max, linestyles='--')
        currAx.set_xlabel("")
        sns.despine(ax=currAx, offset=5, trim=True)
        currAx.tick_params(labelsize=24, rotation=45)
        currAx.set_xlabel("")
        currAx.set_ylabel("")
        currAx.set_title(param)
    plt.tight_layout()

def PlotParameterDistribution_bars(summaryDf, exampleFit, x="PatientId", showAll=False,
                                   nCols=5, palette=None, figsize=(12, 4), ax=None):
    paramNamesList = list(exampleFit.params.keys()) if showAll else exampleFit.var_names
    nParams = len(paramNamesList)

    if ax is None: fig, axList = plt.subplots(ceil(nParams/nCols), nCols, figsize=figsize)
    nPatients = summaryDf.shape[0]
    for i, param in enumerate(paramNamesList):
        currAx = axList.flatten()[i]
        sns.barplot(x=x,y=param, edgecolor=".2", linewidth=2.5, palette=palette, data=summaryDf, ax=currAx)
        currAx.errorbar(x=np.arange(0,summaryDf.shape[0]), y=summaryDf[param],
                 yerr=summaryDf[param+"_se"].values,
                 fmt='none', c='black', capsize=3)
        currAx.hlines(xmin=-1, xmax=nPatients+1, y=exampleFit.params[param].min, linestyles='--')
        currAx.hlines(xmin=-1, xmax=nPatients+1, y=exampleFit.params[param].max, linestyles='--')
        currAx.set_xlabel("")
        currAx.set_ylim(0.75*exampleFit.params[param].min, 1.25*exampleFit.params[param].max)
        sns.despine(ax=currAx, offset=5, trim=True)
        currAx.tick_params(labelsize=24, rotation=45)
        currAx.set_xlabel("")
        currAx.set_ylabel("")
        currAx.set_title(param)
    plt.tight_layout()