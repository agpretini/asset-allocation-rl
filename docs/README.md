# Documentation Map

This directory documents the product contract and engineering decisions. It is
not a copy of generated analysis: reproducible evidence and charts live under
`analysis/`.

| Document | Responsibility |
| --- | --- |
| `PRODUCT_SPEC.md` | Product contract: Model B, decision rule, reward, action space, and evaluation intent. |
| `DATA_DECISIONS.md` | Data universe, frequency, return construction, causality, and simulator data boundaries. |
| `ARCHITECTURE.md` | Module boundaries and the target implementation flow for the Model B product. |
| `DEVELOPEMENT_GUIDELINES.md` | Engineering, test, linting, and contribution conventions. |
| `ROADMAP.md` | Completed milestones, current migration work, and explicitly deferred work. |

When documents overlap, use this precedence: the product decision rule belongs
to `PRODUCT_SPEC.md`; data definitions belong to `DATA_DECISIONS.md`; code
ownership belongs to `ARCHITECTURE.md`. Historical A/C comparisons are retained
only as rationale for selecting B, not as competing product contracts.

## Formulation names

The A/B/C names originated in the controlled state-and-cost diagnostic. They
are shorthand, not versions of the repository:

| Name | Learned state | Cost treatment | Role |
| --- | --- | --- | --- |
| **A — baseline** | Market state plus current portfolio | CRRA reward net of cost; raw `argmax Q` | Historical reference. |
| **B — selected** | Market state only | CRRA reward net of cost; subtract rebalance cost again when scoring an action | Product contract. |
| **C — control** | Market state only | CRRA reward and action selection ignore costs | Frictionless diagnostic control. |

`PRODUCT_SPEC.md` defines B in full. A and C are retained only to explain the
selection evidence and to provide comparison controls.
