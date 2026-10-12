# Units and interpretation of supporting model outputs

The primary normalized gamma index is dimensionless. All other outputs are conditional transformations of that index. They have not been calibrated against independent physiological measurements.

| Code field | Implemented units | Interpretation |
| --- | --- | --- |
| CMRO2 | μmol/(g·min) on an imposed scale | Proxy C = 2.1 + u; numerical difference equals index difference because the selected slope is 1 |
| CBF | mL/(g·min) | Multiply by 100 for mL/(100 g·min); modeled flow from the assumed oxygen balance |
| OEF | Dimensionless fraction | Modeled extraction, not measured |
| ph_V | Dimensionless pH | Modeled venous value |
| p_CO2_V | torr | Modeled venous CO2 partial pressure |
| pO2_cap | torr | Hill-equation capillary oxygen proxy |
| DeltaHCO2 | μmol/(g·min), given implemented concentrations and flow | CO2-associated modeled proton flux; nonzero at selected baseline, despite Delta in its code name |
| DeltaLAC | μmol/(g·min), given implemented concentrations and flow | Lactate-associated modeled flux term |

The unaltered raw fields in simulation_draws.csv and simulation_summary.csv follow these units. The eight fields are not independent observations. Constant provenance and physiological applicability require domain-expert review. A computational balance check cannot establish biological validity.
