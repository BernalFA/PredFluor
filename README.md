# PredFluor: Predicting Fluorescence Emission Wavelengths and Quantum Yields via Machine Learning


High-throughput screening (HTS) campaigns using fluorescence-based readouts, may suffer from auto-fluorescence (AF) as a common interference. Thus, prediction of the fluorescence emission wavelengths (and quantum yields) of organic molecules may represent a key step for identification of potential AF interference. Recently, [Souza et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02403) reported random forest models for the prediction of wavelength and quantum yield (maximum emission) of organic compounds. Based on the limited usefulness of the jupyter-notebooks in the GitHub [repository](https://github.com/Quimica-Teorica-IME/Predicting-Fluorescence-Emission-Wavelengths-and-Quantum-Yields-via-Machine-Learning) accompanying the publication, here we have established a ready-to-use implementation that allows direct prediction of both properties for a given set of SMILES strings.  

&nbsp; 

## Installation

Follow conventional installation using a virtual environment (e.g. using conda):

```bash
# conda environment
conda create -n predfluor python=3.9
conda activate predfluor
# installation
git clone https://github.com/BernalFA/PredFluor.git
cd predfluor
pip install .
```

&nbsp; 

## Usage

Prediction of emission wavelength and the corresponding quantum yield can be achieved using the provided python script, for example:

```bash
python predfluor/prediction_script.py DATA/Only_wl_data_ML.csv -smi_col "Chromophore" -o test_prediction
```

Use `python prediction_script.py -h` to see more information on how to use this script.

&nbsp; 

PredFluor can be also used as a python package.

```python
import pandas as pd
from predfluor import FluorescencePredictor

# read SMILES
df = pd.read_csv("DATA/Only_wl_data_ML.csv")
# Instantiate predictor
predictor = FluorescencePredictor()
# get prediction using same solvent for all the compounds
result = predictor.predict(smiles=df["Chromophore"], solv_smiles=["O"] * len(df))
```

&nbsp; 

## Models

For information about the models' architecture and performance, please check out the original publication.