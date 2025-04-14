import warnings
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from sklearn.exceptions import InconsistentVersionWarning
from utils import FluorescencePredictor

# Silence RDKit deprecation warnings for fingerprint generation
# https://github.com/rdkit/rdkit/issues/2683
RDLogger.DisableLog("rdApp.*")

# Silence sklearn warning caused by differential versioning of scaler and models
# WHY did the authors do that??
# https://stackoverflow.com/questions/29086398/sklearn-turning-off-warnings
warnings.filterwarnings(action="ignore", category=InconsistentVersionWarning)

# Define module path
MOD_PATH = Path(__file__).parent

# main script
if "__main__" == __name__:
    # read dataframes with SMILES and solvents
    df1 = pd.read_csv((MOD_PATH / "../DATA/Only_wl_data_ML.csv").resolve())
    df2 = pd.read_csv((MOD_PATH / "../DATA/Only_qy_data_ML.csv").resolve())
    # Instantiate predictor
    predictor = FluorescencePredictor()
    # Calculate wavelengths and quantum yields
    res1 = predictor.predict(df1["Chromophore"], df1["Solvent"])
    assert df1.shape[0] == res1.shape[0]

    res2 = predictor.predict(df2["Chromophore"], df2["Solvent"])
    assert df2.shape[0] == res2.shape[0]
