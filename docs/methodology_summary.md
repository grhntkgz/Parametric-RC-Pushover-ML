# Methodology Summary

The workflow has four main stages:

1. Generate controlled randomized reinforced concrete building parameters.
2. Filter candidate models using preliminary engineering rules.
3. Build and analyze SAP2000 models in X and Y pushover directions.
4. Extract structural response indicators and evaluate behavior groups using classification-based machine learning.

The machine-learning module is used as an interpretation aid. It identifies which input parameters distinguish selected behavior groups from all remaining observations. It is not intended to replace nonlinear structural analysis.
