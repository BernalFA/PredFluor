import argparse
import warnings
from pathlib import Path

import numpy as np
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


# Helper function for argument parsing
def arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("filepath", help="path to file containing SMILES.")
    parser.add_argument(
        "-smi_col",
        default="smiles",
        help="Name of SMILES column when filepath is Excel or CSV. Default: 'smiles'",
    )
    parser.add_argument(
        "-s",
        dest="solvent",
        default="O",
        help="SMILES string for solvent. Default: 'O' (water)",
    )
    parser.add_argument(
        "-o", dest="output", default="output.csv", help="Base output filename."
    )

    return parser.parse_args()


# main script
if "__main__" == __name__:
    # Get user input
    args = arg_parser()
    # read file
    if args.filepath.endswith(".xlsx"):
        df = pd.read_excel(args.filepath)
    elif args.filepath.endswith(".csv"):
        df = pd.read_csv(args.filepath)
    else:
        raise ValueError(f"File {args.filepath} could not be read.")
    # Instantiate predictor
    predictor = FluorescencePredictor()
    # Calculate wavelengths and quantum yields
    result = predictor.predict(df[args.smi_col], [args.solvent] * len(df))
    # Save to file
    if args.output.endswith(".csv"):
        output = args.output
    else:
        output = args.output + ".csv"
    np.savetxt(output, result, delimiter=",", header="Wavelength, QY")
