# Strobl et al (2024). *To modulate or to skip: De-escalating PARP inhibitor maintenance therapy in ovarian cancer using adaptive therapy*
This repository contains the code and data for our publication Strobl et al (2024). *To modulate or to skip: De-escalating PARP inhibitor maintenance therapy in ovarian cancer using adaptive therapy*, Cell Systems xxx, available [here](xxx) [xxx]. 
A pre-print of our manuscript is available on the [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.22.533721v1) [2].

![Visual summary of our study. Using an integrative and iterative process we developed and validated a mathematical model of the PARPi response in ovarian cancer. Leveraging this model we explored different adaptive therapy algorithm, finding that modulations based strategies are superior to skipping. This is due to a delay in the cell kill and a diminishing dose response. Preliminary in vivo experiments confirm that adaptive modulation can control the tumor as well as continuous therapy whilst halving drug use](visual_abstract.png)

## Requirements
A full list of the Python packages used in this project can be found in `requirements.txt`. To recreate the virtual 
environment, run:
```console
$ conda create --name <envname> --file requirements.txt
$ source <env_name>/bin/activate
``` 
For further details, see [here](https://stackoverflow.com/questions/41249401/difference-between-pip-freeze-and-conda-list)

## Data
Both the raw and processed data files can be found in the `data` folder. These contain all the confluence vs time data used to calibrate and validate the models shown in the paper (in vitro and in vivo). The data processing steps are documented in `jnb_dataProcessing.ipynb`:
- `continuousTreatmentDf_raw.csv` contains the data for the experiments in which we treated cells continuously at different doses and from different starting densities.
- `continuousTreatmentDf_cleaned.csv` contains the cleaned continuous treatment data that we used for model fitting/testing (see `jnb_dataProcessing.ipynb` for details of post-processing).
- `intermittentTreatmentDf_oc3_raw.csv` and `intermittentTreatmentDf_oc3_raw.csv` contains the raw data for the experiments in which we treated cells for some time and then withdrew treatment.
- `intermittentTreatmentDf_cleaned.csv` contains the cleaned intermittent treatment data that we used for model fitting/testing (see `jnb_dataProcessing.ipynb` for details of post-processing).
- `mouseDataDf_oc3.csv` contains the volume vs time data from the in vivo experiment.
- `sweep_mod_vs_skipping.csv` contains the simulation data from comparing modulation-based and skipping based strategies in Figure 7a.

## Analysis
For each results figure in the manuscript we have created a separate jupyter notebook which houses the code to 
re-create this figure. These are named `jnb_figure2.ipynb` etc. and contain further explanations within. 

In case of questions or comments, feel free to [reach out to me](https://stroblmar.github.io/) at anytime.

## References
- ﻿[1] xxx
- ﻿[2] Strobl, M. et al. (2023). Adaptive therapy for ovarian cancer: An integrated approach to PARP inhibitor scheduling. bioRxiv 2023.03.22.533721, doi:10.1101/2023.03.22.533721.