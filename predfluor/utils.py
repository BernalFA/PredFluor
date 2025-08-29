"""
Module containing the FluorescencePredictor class for molecular featurization and
prediction of fluorescence and quantum yield of small molecules (from smiles),
based on the notebooks provided in the original repo.

@author: Dr. Freddy Bernal
"""

import importlib.resources as resources
import warnings
from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    GraphDescriptors,
    Lipinski,
    rdMolDescriptors,
)
from tqdm import tqdm


# Instantiate namedtuples for models and scalers
Models = namedtuple("Models", "wl qy")
Scalers = namedtuple("Scalers", "wl qy")


def flatten_list(lst: list) -> list:
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


class FluorescencePredictor:
    """Class for molecular featurization and prediction of fluorescence wavelength and
    quantum yield of (small) organic molecules (as SMILES strings).
    It uses the models presented in Souza et al. (DOI: 10.1021/acs.jcim.4c02403)

    Example:
        smiles = ["Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                  "c2ccc(CNc1ccccc1)cc2"]
        solvents = ["O", "C1CCCCC1"]

        predictor = FluorescencePredictor()
        results = predictor.predict(smiles, solvents)

    results is a numpy array containing the wavelengths (results[:, 0]) and the quantum
    yields (results[:, 1]) predicted by the trained models.

    IMPORTANT: invalid SMILES are processed as a zero-containing feature vector. Thus,
    they still produce some predicted values. Please check the given warnings in case
    of invalid SMILES.
    """

    def __init__(self):
        self.models = self._load_trained_models()
        self.scalers = self._load_fitted_scalers()

    def _load_trained_models(self) -> namedtuple:
        """Load trained models for QY and WL from models folder

        Returns:
            namedtuple: loaded models (qy and wl)
        """
        model_qy = self._load_pkl("QY_Random.pkl")
        model_wl = self._load_pkl("WL_Random.pkl")
        return Models(model_wl, model_qy)

    def _load_fitted_scalers(self) -> namedtuple:
        """Load fitted scalers to use before QY and WL predictions.

        Returns:
            namedtuple: loaded scalers (qy and wl)
        """
        scaler_qy = self._load_pkl("scaler_model-QY.pkl")
        scaler_wl = self._load_pkl("scaler_modelwl.pkl")
        return Scalers(scaler_wl, scaler_qy)

    def _load_pkl(self, file: str):
        """Helper to load any model or scaler from package resources

        Args:
            file (str): model or scaler file name (dumped).
        """
        with resources.files(__package__).joinpath(f"models/{file}").open("rb") as f:
            model = joblib.load(f)
        return model

    def _get_morgan_fp(self, m: Chem.Mol) -> np.ndarray:
        """Create Morgan fingerprint for radius 2 (512 bits) for given molecule.

        Args:
            m (Chem.Mol): RDKit mol object.

        Returns:
            np.ndarray: generated fingerprint (len = 512).
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 512)
        fp = np.array(list(fp), dtype=np.int16)
        return fp

    def _get_maccs_keys(self, m: Chem.Mol) -> np.ndarray:
        """Create MACCS fingerprint for given molecule.

        Args:
            m (Chem.Mol): RDKit mol object.

        Returns:
            np.ndarray: generated fingerprint.
        """
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(m)
        fp = np.array(list(fp), dtype=np.int16)
        return fp

    def _get_rdkit_descriptors(self, m: Chem.Mol) -> np.ndarray:
        """Calculate RDKit selected descriptors for given molecule. Some descriptor
        are scaled by default.

        Args:
            m (Chem.Mol): RDKit mol object.

        Returns:
            np.ndarray: calculated descriptors (len = 39).
        """
        desc = [
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
        desc = np.array(desc, dtype=np.int16)
        return desc

    def _get_lipinski_desc(self, m: Chem.Mol) -> np.ndarray:
        """Calculate Lipinski's rule of five related and graph-based descriptors for
        given molecule.

        Args:
            m (Chem.Mol): RDKit mol object.

        Returns:
            np.ndarray: calculated descriptors (len = 7).
        """
        desc = [
            Lipinski.HeavyAtomCount(m),
            Lipinski.NHOHCount(m),
            Lipinski.NOCount(m),
            Lipinski.NumHAcceptors(m),
            Lipinski.NumHDonors(m),
            GraphDescriptors.Chi0(m),
            GraphDescriptors.Chi1(m),
        ]
        desc = np.array(desc, dtype=np.int16)
        return desc

    def _get_bcut2d_desc(self, m: Chem.Mol) -> np.ndarray:
        """Calculate BTCU descriptors for given molecule (as implemented in RDKit).

        Args:
            m (Chem.Mol): RDKit mol object.

        Returns:
            np.ndarray: calculated descriptors.
        """
        desc = [rdMolDescriptors.BCUT2D(m)]
        desc = np.array(flatten_list(desc), dtype=np.int16)
        return desc

    def get_features(self, smiles: Iterable[str]) -> np.ndarray:
        """Calculate all the necessary molecular features for prediction of fluorescence
        wavelength and quantum yield for given .

        Args:
            smiles (Iterable[str]): SMILES strings for molecules to be predicted.

        Returns:
            np.ndarray: concatenated set of molecular features (len = 733).
        """
        # Iterate over molecules
        features_list = []
        warning_list = []
        for i, smi in enumerate(tqdm(smiles, desc="Processing SMILES")):
            m = Chem.MolFromSmiles(smi)
            # Calculate features for valid SMILES
            if m is not None:
                feat1 = self._get_morgan_fp(m)
                feat2 = self._get_maccs_keys(m)
                feat3 = self._get_rdkit_descriptors(m)
                feat4 = self._get_lipinski_desc(m)
                feat5 = self._get_bcut2d_desc(m)
                features = np.concatenate([feat1, feat2, feat3, feat4, feat5])
            else:
                warning_list.append(i)
                # add a zeros containing array for invalid SMILES
                features = np.zeros((733))  # 733 is the number of features
            features_list.append(features)
        if warning_list:
            n = len(warning_list)
            # Display warning for invalid SMILES
            msg = f"\nThe following SMILES ({n}) could not be converted into RDKit Mol"
            warnings.warn(msg)
            for i in warning_list:
                print(f"{smiles[i]} at index {i}")

        return np.array(features_list)

    def _calculate_features(
        self, smiles: Iterable[str], solv_smiles: Iterable[str]
    ) -> np.ndarray:
        """calculate a combined molecular features vector for the molecule and the
        corresponding solvent. It uses get_features.

        Args:
            smiles (Iterable[str]): SMILES string for molecules of interest.
            solv_smiles (Iterable[str]): SMILES string for solvents (the number of
                                         solvents must be equal to the number of
                                         compounds).

        Returns:
            np.ndarray: concatenated features.
        """
        features_cmpd = self.get_features(smiles)
        features_solv = self.get_features(solv_smiles)
        features = np.hstack([features_cmpd, features_solv])
        return features

    def _predict_wavelength(self, features: np.ndarray) -> np.ndarray:
        """Calculate fluorescence wavelength from given set of molecular features.

        Args:
            features (np.ndarray): combined molecular features for compound and solvent
                                   obtained with _calculate_features.

        Returns:
            np.ndarray: predicted wavelength of maximum fluorescence.
        """
        features_scaled_wl = self.scalers.wl.transform(features)
        predicted_wl = self.models.wl.predict(features_scaled_wl)
        return predicted_wl

    def _predict_quantum_yield(self, features: np.ndarray) -> np.ndarray:
        """Calculate quantum yield from given set of molecular features.

        Args:
            features (np.ndarray): combined molecular features for compound and solvent
                                   obtained with _calculate_features.

        Returns:
            np.ndarray: predicted quantum yield of maximum fluorescence.
        """
        features_scaled_qy = self.scalers.qy.transform(features)
        predicted_qy = self.models.qy.predict(features_scaled_qy)
        return predicted_qy

    def predict(self, smiles: Iterable[str], solv_smiles: Iterable[str]) -> np.ndarray:
        """Use trained models to predict the wavelength and quantum yield of maximum
        fluorescence of (small) organic molecules for given compounds in defined
        solvents (as SMILES strings).

        Args:
            smiles (Iterable[str]): SMILES strings for query compounds.
            solv_smiles (Iterable[str]): SMILES strings for solvents disolving the
                                         compounds.

        Returns:
            np.ndarray: predicted wavelength and quantum yield (first and second column,
                        respectively).
        """
        features = self._calculate_features(smiles, solv_smiles)
        predicted_wl = self._predict_wavelength(features)
        predicted_qy = self._predict_quantum_yield(features)
        result = np.stack((predicted_wl, predicted_qy), axis=1)
        return result
