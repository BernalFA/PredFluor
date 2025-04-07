"""
Module for featurization and prediction of fluorescence based on notebooks in original
repo.

@author: Dr. Freddy Bernal
"""

from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import (AllChem, Descriptors, GraphDescriptors, Lipinski,
                        rdMolDescriptors)


MOD_PATH = Path(__file__).parent


def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def properties_array(sSmiles):
    try:
        m = Chem.MolFromSmiles(sSmiles)
        p1 = AllChem.GetMorganFingerprintAsBitVect(m, 2, 512)
        p2 = rdMolDescriptors.GetMACCSKeysFingerprint(m)

        p3 = [
            1000 * Descriptors.FpDensityMorgan1(m),
            1000 * Descriptors.FpDensityMorgan2(m),
            1000 * Descriptors.FpDensityMorgan3(m),
            Descriptors.ExactMolWt(m),
            Descriptors.HeavyAtomMolWt(m),
            1000 * Descriptors.MaxAbsPartialCharge(m),
            1000 * Descriptors.MaxPartialCharge(m),
            1000 * Descriptors.MinAbsPartialCharge(m),
            1000 * Descriptors.MinPartialCharge(m),
            Descriptors.NumRadicalElectrons(m),
            Descriptors.NumValenceElectrons(m),
            1000 * rdMolDescriptors.CalcFractionCSP3(m),
            10 * rdMolDescriptors.CalcKappa1(m),
            10 * rdMolDescriptors.CalcKappa2(m),
            10 * rdMolDescriptors.CalcKappa3(m),
            rdMolDescriptors.CalcLabuteASA(m),
            rdMolDescriptors.CalcNumAliphaticCarbocycles(m),
            rdMolDescriptors.CalcNumAliphaticHeterocycles(m),
            rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumAmideBonds(m),
            rdMolDescriptors.CalcNumAromaticCarbocycles(m),
            rdMolDescriptors.CalcNumAromaticHeterocycles(m),
            rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcNumAtomStereoCenters(m),
            rdMolDescriptors.CalcNumBridgeheadAtoms(m),
            rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumHeteroatoms(m),
            rdMolDescriptors.CalcNumHeterocycles(m),
            rdMolDescriptors.CalcNumLipinskiHBA(m),
            rdMolDescriptors.CalcNumLipinskiHBD(m),
            rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumRotatableBonds(m),
            rdMolDescriptors.CalcNumSaturatedCarbocycles(m),
            rdMolDescriptors.CalcNumSaturatedHeterocycles(m),
            rdMolDescriptors.CalcNumSaturatedRings(m),
            rdMolDescriptors.CalcNumSpiroAtoms(m),
            rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(m),
            rdMolDescriptors.CalcTPSA(m),
        ]
        pa3 = np.array(p3, dtype=np.int16)

        p4 = [
            Lipinski.HeavyAtomCount(m),
            Lipinski.NHOHCount(m),
            Lipinski.NOCount(m),
            Lipinski.NumHAcceptors(m),
            Lipinski.NumHDonors(m),
            GraphDescriptors.Chi0(m),
            GraphDescriptors.Chi1(m),
        ]

        p5 = [rdMolDescriptors.BCUT2D(m)]

        pa1 = np.array(list(p1), dtype=np.int16)
        pa2 = np.array(list(p2), dtype=np.int16)
        pa0 = np.concatenate([pa1, pa2])
        pa4 = np.array(p4, dtype=np.int16)
        pa5 = np.array(flatten_list(p5), dtype=np.int16)

        pa = np.concatenate([pa0, pa3, pa4, pa5])

        pa = np.array(pa)

        return pa, True
    except Exception:
        # print(f"Ocorreu um erro: {e}")
        return None, False


def predict_fluorescence(smiles, solv_smi="O"):

    model_qy = joblib.load((MOD_PATH / "../Models/QY_Random.pkl").resolve())
    model_wl = joblib.load((MOD_PATH / "../Models/WL_Random.pkl").resolve())
    scaler_wl = joblib.load((MOD_PATH / "../Models/scaler_modelwl.pkl").resolve())
    scaler_qy = joblib.load((MOD_PATH / "../Models/scaler_model-QY.pkl").resolve())

    properties, valid = properties_array(smiles)
    prop_solv, valid_solv = properties_array(solv_smi)

    if valid and valid_solv:
        features = np.concatenate([properties, prop_solv]).reshape(1, -1)
        features_scaled_qy = scaler_qy.transform(features)
        features_scaled_wl = scaler_wl.transform(features)

        predicted_qy = model_qy.predict(features_scaled_qy)[0]
        predicted_wl = model_wl.predict(features_scaled_wl)[0]

        return predicted_wl, predicted_qy

    else:
        raise ValueError(f"{smiles=} invalid")
