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
from lmfit import minimize
from tqdm import tqdm
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sys.path.append("./")
import myUtils as utils
from odeModels import MakeModelFromStr

# ====================================================================================
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

# ====================================================================================
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

# ====================================================================================
def PerturbParams(params):
    params = params.copy()
    for p in params.keys():
        currParam = params[p]
        if currParam.vary:
            params[p].value = np.random.uniform(low=currParam.min, high=currParam.max)
    return params

# ====================================================================================
def ComputeRSquared(fit,dataDf,feature="Confluence"):
    tss = np.sum(np.square(dataDf[feature]-dataDf[feature].mean()))
    rss = np.sum(np.square(fit.residual))
    return 1-rss/tss

# ====================================================================================
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

# ====================================================================================
def PlotFit(fitObj, dataDf, model=None, dt=1, linewidth=5, linewidthA=5, titleStr="", legend=True, outName=None, ax=None, solver_kws={}, **kwargs):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if model is None:
        myModel = MakeModelFromStr(fitObj.modelName)
        myModel.SetParams(**fitObj.params.valuesdict())
    else:
        myModel = model
    solver_kws['max_step'] = solver_kws.get('max_step',1) # Use fine-grained time-stepping unless otherwise specified
    myModel.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(dataDf),**solver_kws)
    myModel.Trim(dt=dt)
    myModel.Plot(ymin=0, title=titleStr, linewidth=linewidth, linewidthA=linewidthA, ax=ax, plotLegendB=legend, **kwargs)
    PlotData(dataDf, plotDrugConcentration=False, ax=ax, **kwargs)
    if outName is not None: plt.savefig(outName); plt.close()

# ====================================================================================
def LoadFit(modelName, fitId=0, fitDir="./", model=None, **kwargs):
    fitObj = pickle.load(open(os.path.join(fitDir, "fitObj_fit_%d.p"%(fitId)), "rb"))
    myModel = MakeModelFromStr(modelName, **kwargs) if model is None else model
    myModel.SetParams(**fitObj.params.valuesdict())
    return fitObj, myModel

# ====================================================================================
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

# ====================================================================================
def perform_bootstrap(fitObj, n_bootstraps=5, shuffle_params=True, show_progress=True, outName=None, **kwargs):
    '''
    Function to estimate uncertainty in the parameter estimates and model predictions using a
    parametric bootstrapping method. This means, it uses the maximum likelihood estimate (best fit
    based on least squared method) to generate n_bootstrap synthetic data sets (noise is generated
    by drawing from an error distribution N(0,sqrt(ssr/df))). Subsequently it fits to this synthetic
    data to obtain a distribution of parameter estimates (one estimate/prediction
    per synthetic data set).
    '''
    # Initialise
    nvarys = fitObj.nvarys
    residual_variance = np.sum(np.square(fitObj.residual)) / fitObj.nfree
    paramsToEstimateList = [param for param in fitObj.params.keys() if fitObj.params[param].vary]

    # 1. Perform bootstrapping
    parameterEstimatesMat = np.zeros((n_bootstraps, nvarys+1))  # Array to hold parameter estimates for CI estimation
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # i) Generate synthetic data by sampling from the error model (assuming a normal error distribution)
        tmpDataDf = fitObj.data.copy()
        bestFitPrediction = tmpDataDf['Confluence'] - fitObj.residual
        tmpDataDf['Confluence'] = bestFitPrediction + np.random.normal(loc=0, scale=np.sqrt(residual_variance),
                                                                       size=fitObj.ndata)
        # ii) Fit to synthetic data
        if fitObj.params['N0'].vary == False: tmpDataDf.loc[0, 'Confluence'] = fitObj.data.Confluence.iloc[
            0]  # Remove variation in initial synthetic data if not fitting initial conditions;
        # otherwise this will blow up the residual variance as no fit can ever do well on the IC
        currParams = fitObj.params.copy()
        if shuffle_params:  # Generate a random initial parameter guess
            for param in paramsToEstimateList:
                currParams[param].value = np.random.uniform(low=currParams[param].min,
                                                            high=currParams[param].max)
        # Fit
        tmpModel = MakeModelFromStr(fitObj.modelName)
        currFitObj = minimize(residual, currParams, args=(0, tmpDataDf, tmpModel,
                                                          "Confluence", kwargs.get('solver_kws', {})),
                              **kwargs.get('optimiser_kws', {}))
        # Record parameter estimates for CI estimation
        for i, param in enumerate(paramsToEstimateList):
            parameterEstimatesMat[bootstrapId, i] = currFitObj.params[param].value
        parameterEstimatesMat[bootstrapId, -1] = np.sum(np.square(currFitObj.residual))
    # Return results
    resultsDf = pd.DataFrame(parameterEstimatesMat, columns=paramsToEstimateList+['SSR'])
    if outName is not None: resultsDf.to_csv(outName)
    return resultsDf

# ====================================================================================
def compute_confidenceInterval_prediction(fitObj, bootstrapResultsDf, alpha=0.95,
                                          treatmentScheduleList=None, initialConditionsList=None,
                                          t_eval=None, n_time_steps=100,
                                          show_progress=True, **kwargs):
    # Initialise
    t_eval = np.linspace(fitObj.data.Time.min(), fitObj.data.Time.max(), n_time_steps) if t_eval is None else t_eval
    n_timePoints = len(t_eval)
    treatmentScheduleList = treatmentScheduleList if treatmentScheduleList is not None else utils.ExtractTreatmentFromDf(
        fitObj.data)
    n_bootstraps = bootstrapResultsDf.shape[0]

    # 1. Perform bootstrapping
    modelPredictionsMat_mean = np.zeros(
        (n_bootstraps, n_timePoints))  # Array to hold model predictions for CI estimation
    modelPredictionsMat_indv = np.zeros(
        (n_bootstraps, n_timePoints))  # Array to hold model predictions with residual variance for PI estimation
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # Set up the model using the parameters from a bootstrap fit
        tmpModel = MakeModelFromStr(fitObj.modelName)
        currParams = fitObj.params.copy()
        for var in bootstrapResultsDf.columns:
            if var == "SSR": continue
            currParams[var].value = bootstrapResultsDf[var].iloc[bootstrapId]
        tmpModel.SetParams(**currParams)
        # Calculate confidence intervals for model prediction
        if initialConditionsList is not None: tmpModel.SetParams(**initialConditionsList)
        tmpModel.Simulate(treatmentScheduleList=treatmentScheduleList,
                          solver_kws=kwargs.get('solver_kws', {}))
        tmpModel.Trim(t_eval=t_eval)
        residual_variance_currEstimate = bootstrapResultsDf['SSR'].iloc[
                                             bootstrapId] / fitObj.nfree  # XXX Watch this when extending to treatment model XXX
        modelPredictionsMat_mean[bootstrapId, :] = tmpModel.resultsDf.TumourSize.values
        modelPredictionsMat_indv[bootstrapId, :] = tmpModel.resultsDf.TumourSize.values + np.random.normal(loc=0,
                                                                                                           scale=np.sqrt(
                                                                                                               residual_variance_currEstimate),
                                                                                                           size=n_timePoints)

    # 3. Estimate confidence and prediction interval for model prediction
    tmpDicList = []
    # Compute the model prediction for the model with the MLE parameter estimates
    tmpModel.SetParams(**fitObj.params)  # Calculate model prediction for best fit
    if initialConditionsList is not None: tmpModel.SetParams(**initialConditionsList)
    if treatmentScheduleList is None: treatmentScheduleList = utils.ExtractTreatmentFromDf(fitObj.data)
    tmpModel.Simulate(treatmentScheduleList=treatmentScheduleList,
                      solver_kws=kwargs.get('solver_kws', {}))
    tmpModel.Trim(t_eval=t_eval)
    for i, t in enumerate(t_eval):
        tmpDicList.append({"Time": t, "Estimate_MLE": tmpModel.resultsDf.TumourSize.iloc[i],
                           "CI_Lower_Bound": np.percentile(modelPredictionsMat_mean[:, i], (1 - alpha) * 100 / 2),
                           "CI_Upper_Bound": np.percentile(modelPredictionsMat_mean[:, i],
                                                           (alpha + (1 - alpha) / 2) * 100),
                           "PI_Lower_Bound": np.percentile(modelPredictionsMat_indv[:, i], (1 - alpha) * 100 / 2),
                           "PI_Upper_Bound": np.percentile(modelPredictionsMat_indv[:, i],
                                                           (alpha + (1 - alpha) / 2) * 100)})
    modelPredictionDf = pd.DataFrame(tmpDicList)

    return modelPredictionDf

# ====================================================================================
def benchmark_prediction_accuracy(fitObj, bootstrapResultsDf, dataDf, initialConditionsList=None,
                                  show_progress=True, **kwargs):
    # Initialise
    n_bootstraps = bootstrapResultsDf.shape[0]

    # Compute the r2 value for each bootstrap
    tmpDicList = []
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # Set up the model using the parameters from a bootstrap fit
        tmpModel = MakeModelFromStr(fitObj.modelName)
        currParams = fitObj.params.copy()
        for var in bootstrapResultsDf.columns:
            if var == "SSR": continue
            currParams[var].value = bootstrapResultsDf[var].iloc[bootstrapId]
        if initialConditionsList is not None:
            for var in initialConditionsList.keys():
                currParams[var].value = initialConditionsList[var]

        # Make prediction and compare to true data
        tmpModel.residual = residual(data=dataDf, model=tmpModel, params=currParams,
                                  x=None, feature="Confluence", solver_kws=kwargs.get('solver_kws', {}))
        r2Val = ComputeRSquared(fit=tmpModel, dataDf=dataDf, feature="Confluence")

        # Save results
        tmpDicList.append({"Model":fitObj.modelName, "BootstrapId":bootstrapId,
                           "rSquared":r2Val})
    return pd.DataFrame(tmpDicList)

# ====================================================================================
def compute_confidenceInterval_parameters(fitObj, bootstrapResultsDf, alpha=0.95):
    # Initialise
    paramsToEstimateList = [param for param in fitObj.params.keys() if fitObj.params[param].vary]

    # Estimate confidence intervals for parameters from bootstraps
    tmpDicList = []
    for i, param in enumerate(paramsToEstimateList):
        tmpDicList.append({"Parameter": param, "Estimate_MLE": fitObj.params[param].value,
                           "Lower_Bound": np.percentile(bootstrapResultsDf[param].values, (1 - alpha) * 100 / 2),
                           "Upper_Bound": np.percentile(bootstrapResultsDf[param].values,
                                                        (alpha + (1 - alpha) / 2) * 100)})
    return pd.DataFrame(tmpDicList)