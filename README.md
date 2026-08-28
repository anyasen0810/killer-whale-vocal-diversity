# Using Bioacoustic and Genetic Data to Test the Fidelity of Vertical Vocal Transmission across Killer Whale Subpopulations  

Code for  'Using Bioacoustic and Genetic Data to Test the Fidelity of Vertical Vocal Transmission across Killer Whale Subpopulations'
comparing acoustic distance (via Perch 2.0 embeddings) against mitochondrial and nuclear genetic distance across seven eastern North 
Pacific killer whale subpopulations.

## Pipeline (see `final_pipeline/`)
1. `download_files.py` — downloads audio from the DCLDE dataset (Palmer et al., 2025)
2. `preprocessing_and_exploration.py` — bandpass filtering, resampling, trimming 
3. `feature_extraction.py` — Perch 2.0 embedding generation
4. `embeddings_evaluation.py` — random forest classifier evaluation
5. `UMAP.py` — dimensionality reduction for visualisation
6. `mmd_matrix.py`, `mtDNA_matrix.py`, `mtdna_lengths.py` — distance matrix construction
7. `mantel_tests.py` — Spearman Mantel tests
8. `data_visualisation.py` — figure generation

`archive/` contains earlier development versions, superseded by `final_pipeline/`.

## Requirements
See `requirements.txt`. Install with `pip install -r requirements.txt`.

## Data
Audio data is sourced from the DCLDE 2026 dataset (Palmer et al., 2025) 
and is not included in this repository. 
