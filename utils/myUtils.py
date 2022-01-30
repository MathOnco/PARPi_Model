# ====================================================================================
# Various functions that I found useful in this project
# ====================================================================================
import numpy as np
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import datetime
import os

# ====================================================================================
# Functions for dealing with treatment schedules
# ====================================================================================
# Helper function to extract the treatment schedule from the data
def ConvertTDToTSFormat(timeVec,drugIntensityVec):
    treatmentScheduleList = [] # Time intervals in which we have the same amount of drug
    tStart = timeVec[0]
    currDrugIntensity = drugIntensityVec[0]
    for i,t in enumerate(timeVec):
        if drugIntensityVec[i]!=currDrugIntensity and not (np.all(np.isnan([drugIntensityVec[i],currDrugIntensity]))): # Check if amount of drug has changed
            treatmentScheduleList.append([tStart,t,currDrugIntensity])
            tStart = t
            currDrugIntensity = drugIntensityVec[i]
    treatmentScheduleList.append([tStart,timeVec[-1]+(tStart==timeVec[-1])*1,currDrugIntensity])
    return treatmentScheduleList

# Helper function to obtain treatment schedule from calibration data
def ExtractTreatmentFromDf(dataDf,timeColumn="Time",treatmentColumn="DrugConcentration"):
    timeVec = dataDf[timeColumn].values
    nDaysPreTreatment = int(timeVec.min())
    if nDaysPreTreatment != 0: # Add the pretreatment phase if it's not already added
        timeVec = np.concatenate((np.arange(0, nDaysPreTreatment), timeVec), axis=0)
    drugIntensityVec = dataDf[treatmentColumn].values
    drugIntensityVec = np.concatenate((np.zeros((nDaysPreTreatment,)), drugIntensityVec), axis=0)
    return ConvertTDToTSFormat(timeVec, drugIntensityVec)

# Turns a treatment schedule in list format (i.e. [tStart, tEnd, DrugConcentration]) into a time series
def TreatmentListToTS(treatmentList,tVec):
    drugConcentrationVec = np.zeros_like(tVec)
    for drugInterval in treatmentList:
        drugConcentrationVec[(tVec>=drugInterval[0]) & (tVec<=drugInterval[1])] = drugInterval[2]
    return drugConcentrationVec

# Extract the date as a datetime object from a model or experiment data frame
def GetDateFromDataFrame(df):
    year, month, day, hour, minute = [df[key].values[0] for key in ['Year','Month','Day','Hour','Minute']]
    hour = 12 if np.isnan(hour) else hour
    minute = 0 if np.isnan(minute) else minute
    return datetime.datetime(int(year),int(month),int(day),int(hour),int(minute))

# ====================================================================================
# Misc
# ====================================================================================
def mkdir(dirName):
    """
    Recursively generate a directory or list of directories. If directory already exists be silent. This is to replace
    the annyoing and cumbersome os.path.mkdir() which can't generate paths recursively and throws errors if paths
    already exist.
    :param dirName: if string: name of dir to be created; if list: list of names of dirs to be created
    :return: Boolean
    """
    dirToCreateList = [dirName] if type(dirName) is str else dirName
    for directory in dirToCreateList:
        currDir = ""
        for subdirectory in directory.split("/"):
            currDir = os.path.join(currDir, subdirectory)
            try:
                os.mkdir(currDir)
            except:
                pass
        return True