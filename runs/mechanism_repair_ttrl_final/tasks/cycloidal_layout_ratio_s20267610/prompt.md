# Cycloidal reducer layout ratio

Design a single-stage cycloidal reducer layout with 10 fixed ring pins and target reduction ratio 9:1.

* Required topology: fixed housing/ring, eccentric input, cycloidal disc, and output carrier.
* Required ports: `input_port` and `output_port`, both grounded revolute_joint ports.
* Declare `params.ring_pin_count = 10` and `params.declared_ratio = 9`.
* Declare `params.eccentricity_mm = 1.287`.
