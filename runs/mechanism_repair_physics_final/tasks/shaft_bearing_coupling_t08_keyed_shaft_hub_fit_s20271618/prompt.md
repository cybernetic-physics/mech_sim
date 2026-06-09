# Keyed shaft–hub fit

Design a keyed shaft (Ø12.993 mm bore) with keyway width 6.314 mm.

* Set `params.shaft_diameter_mm` = 12.993.
* Set `params.keyway_width_mm` = 6.314.
* Required ports: `hub_face` (frame, grounded), `output_port` (revolute_joint).
* Mobility = 1 (one revolute joint).

## MechanismRepair-Physics verifier contract

This task counts only if the submitted mechanism preserves topology, ports, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics. Fake contact-oracle outputs are not accepted for headline evaluation.
