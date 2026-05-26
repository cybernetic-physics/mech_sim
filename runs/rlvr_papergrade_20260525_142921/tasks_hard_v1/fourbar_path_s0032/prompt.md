# Four-bar coupler path

Design a planar 4-bar mechanism whose coupler point traces the curve in `fixtures/target_path.csv`.

Required ports: `input_port` (revolute_joint, grounded), `output_port` (revolute_joint, grounded), `coupler_point` (frame on the coupler).

Mobility must equal 1. The reference comparison is the symmetric Chamfer distance after centroid+RMS-radius normalization.
