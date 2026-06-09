# Spur gear ratio (analytic)

Design a two-gear reducer with pinion (14 teeth) and gear (56 teeth).

* Declare `params.declared_ratio` = teeth_out / teeth_in = 4.0.
* Required ports: `input_port` (revolute_joint), `output_port` (revolute_joint), both grounded.
* Mobility = 2 (two free axes, ungeared in this analytic tier).

## MechanismRepair-Physics verifier contract

This task counts only if the submitted mechanism preserves topology, ports, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics. Fake contact-oracle outputs are not accepted for headline evaluation.
