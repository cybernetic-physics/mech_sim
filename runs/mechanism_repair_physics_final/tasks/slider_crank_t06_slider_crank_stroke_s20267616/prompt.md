# Slider-crank stroke

Design a centric slider-crank with crank length 27.73 mm and coupler 79.47 mm.

* Declare `params.declared_stroke_mm` = 55.46 mm (twice the crank length).
* Required ports: `input_port` (revolute_joint, grounded), `output_port` (prismatic_joint).
* Mobility = 1.

## MechanismRepair-Physics verifier contract

This task counts only if the submitted mechanism preserves topology, ports, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics. Fake contact-oracle outputs are not accepted for headline evaluation.
