# Cycloidal reducer layout ratio

Design a single-stage cycloidal reducer layout with 14 fixed ring pins and target reduction ratio 13:1.

* Required topology: fixed housing/ring, eccentric input, cycloidal disc, and output carrier.
* Required ports: `input_port` and `output_port`, both grounded revolute_joint ports.
* Declare `params.ring_pin_count = 14` and `params.declared_ratio = 13`.
* Declare `params.eccentricity_mm = 1.151`.
