"""
Cancer type / tissue metadata vocabulary and encoding utilities.

The five cancer types covered are drawn from published spatial transcriptomics
datasets used in computational pathology benchmarks:

  - CCRCC  : Clear Cell Renal Cell Carcinoma (kidney)
  - COAD   : Colon Adenocarcinoma
  - LUAD   : Lung Adenocarcinoma
  - PAAD   : Pancreatic Adenocarcinoma
  - PRAD   : Prostate Adenocarcinoma

Usage
-----
    from histomoe.data.metadata_utils import cancer_type_to_id, id_to_cancer_type
    idx = cancer_type_to_id("CCRCC")  # -> 0
    name = id_to_cancer_type(0)       # -> "CCRCC"
"""

from typing import Dict, List, Optional

# Ordered list of supported cancer types (index == expert ID in MoE)
CANCER_TYPES: List[str] = ["CCRCC", "COAD", "LUAD", "PAAD", "PRAD"]

# Human-readable descriptions per cancer type
CANCER_DESCRIPTIONS: Dict[str, str] = {
    "CCRCC": "Clear Cell Renal Cell Carcinoma — kidney cancer with clear cytoplasm",
    "COAD": "Colon Adenocarcinoma — colorectal cancer originating in glandular cells",
    "LUAD": "Lung Adenocarcinoma — most common lung cancer subtype",
    "PAAD": "Pancreatic Adenocarcinoma — aggressive ductal pancreatic cancer",
    "PRAD": "Prostate Adenocarcinoma — most common prostate malignancy",
}

# Associated organ/tissue for each cancer type
CANCER_TISSUE: Dict[str, str] = {
    "CCRCC": "kidney",
    "COAD": "colon",
    "LUAD": "lung",
    "PAAD": "pancreas",
    "PRAD": "prostate",
}

# Vocabulary: cancer type string -> integer index
_CANCER_TO_ID: Dict[str, int] = {ct: i for i, ct in enumerate(CANCER_TYPES)}
_ID_TO_CANCER: Dict[int, str] = {i: ct for i, ct in enumerate(CANCER_TYPES)}


def cancer_type_to_id(cancer_type: str) -> int:
    """Convert a cancer type string to its integer expert ID.

    Parameters
    ----------
    cancer_type : str
        One of: ``"CCRCC"``, ``"COAD"``, ``"LUAD"``, ``"PAAD"``, ``"PRAD"``.

    Returns
    -------
    int
        Integer expert index (0-indexed).

    Raises
    ------
    ValueError
        If the cancer type is not in the supported vocabulary.
    """
    cancer_type = cancer_type.upper().strip()
    if cancer_type not in _CANCER_TO_ID:
        raise ValueError(
            f"Unknown cancer type '{cancer_type}'. "
            f"Supported types: {CANCER_TYPES}"
        )
    return _CANCER_TO_ID[cancer_type]


def id_to_cancer_type(idx: int) -> str:
    """Convert an expert integer index back to the cancer type string.

    Parameters
    ----------
    idx : int
        Expert index (0-indexed, in range [0, num_experts)).

    Returns
    -------
    str
        Cancer type string.

    Raises
    ------
    ValueError
        If the index is out of range.
    """
    if idx not in _ID_TO_CANCER:
        raise ValueError(
            f"Index {idx} out of range. Valid range: [0, {len(CANCER_TYPES)-1}]"
        )
    return _ID_TO_CANCER[idx]


def build_metadata_string(
    cancer_type: str,
    tissue: Optional[str] = None,
    include_description: bool = False,
) -> str:
    """Build a natural-language metadata string for the text encoder.

    Parameters
    ----------
    cancer_type : str
        Cancer type identifier (e.g., ``"CCRCC"``).
    tissue : str, optional
        Tissue/organ name. If None, inferred from ``CANCER_TISSUE``.
    include_description : bool
        If True, append the longer human-readable cancer description.

    Returns
    -------
    str
        Formatted metadata string.

    Examples
    --------
    >>> build_metadata_string("CCRCC")
    "Cancer type: CCRCC. Tissue: kidney."
    >>> build_metadata_string("CCRCC", include_description=True)
    "Cancer type: CCRCC. Tissue: kidney. Clear Cell Renal Cell Carcinoma..."
    """
    cancer_type = cancer_type.upper().strip()
    tissue = tissue or CANCER_TISSUE.get(cancer_type, "unknown")
    text = f"Cancer type: {cancer_type}. Tissue: {tissue}."
    if include_description and cancer_type in CANCER_DESCRIPTIONS:
        text += f" {CANCER_DESCRIPTIONS[cancer_type]}"
    return text


def num_cancer_types() -> int:
    """Return the total number of supported cancer types (= number of experts)."""
    return len(CANCER_TYPES)
