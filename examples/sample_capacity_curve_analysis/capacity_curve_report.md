# Capacity Curve Derived Metrics

- Metadata source: `examples\sample_metadata`
- Directional observations: 50
- Model-level X/Y pairs: 25
- Envelope-derived monotonic estimate share: 100.00%

## Overall Indicators

- Mean peak base shear: 11518.6 kN
- Mean normalized energy: 0.755
- Mean ductility proxy: 10.032
- Mean X/Y peak-shear asymmetry ratio: 1.032
- Mean X/Y initial-stiffness asymmetry ratio: 1.224

## Damage-Class Summary

| critical_state | count | share | avg_peak_base_shear_kn | avg_initial_stiffness_kn_per_m | avg_normalized_energy | avg_ductility_proxy |
| --- | --- | --- | --- | --- | --- | --- |
| LS-CP | 41 | 0.82 | 11686.4 | 102337 | 0.768448 | 11.4214 |
| IO-LS | 6 | 0.12 | 12626.2 | 92221.6 | 0.733208 | 4.30661 |
| B-IO | 3 | 0.06 | 7010.6 | 44407.8 | 0.608025 | 2.48982 |

## Critical-Element-Type Summary

| critical_type | count | share | avg_peak_base_shear_kn | avg_initial_stiffness_kn_per_m | avg_normalized_energy | avg_ductility_proxy |
| --- | --- | --- | --- | --- | --- | --- |
| column | 50 | 1 | 11518.6 | 97647.2 | 0.754594 | 10.0317 |

## Manuscript Note

These metrics convert individual pushover curves into comparable response indicators. Because some SAP2000 outputs are stored as envelope-derived monotonic estimates rather than full saved nonlinear state histories, the curve source type should be reported when interpreting energy, ductility, and post-peak behavior.
