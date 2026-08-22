# E3 Preference-BO Redundant-Bank Pilot Results

Profile: `smoke`

Configuration SHA-256: `5c53c8de7144d1d0bc3a66b1ca4233b2c17811f726e977e1a5c90f103140b260`

Starting commit: `b68525952f58693461ac32d4658a8d611a594706`

## Scientific gates

Not evaluated: this was the reduced numerical smoke test.

## Mechanical status

All required mechanical checks passed: `True`.

## Numerical caveat

seed=0 full iteration=1: full-target ESS/split diagnostic remained unreliable at the frozen cap; seed=0 adaptive iteration=1: held-out full-target diagnostic remained unreliable at the frozen cap

## Interpretation

Reduced-count smoke execution was mechanically correct; it is not scientific evidence for P1 or P2.


## Next action

Run the unchanged full 12-seed frozen pilot.
