# Robust MILP benchmark variants (Phase 6)

All variants use the same information set, gate closures,
transaction costs, deviation penalties, and physical constraints
as the classic information-equivalent MILP; only the planning
objective changes. RL composites shown for reference. Regret is
vs the rule-based controller on the same 92 validation days.

|                    |   mean |   median |   cvar_regret |   mean_regret |   downside_exposure |
|:-------------------|-------:|---------:|--------------:|--------------:|--------------------:|
| ensemble_mean      |  55635 |    56916 |         -1822 |           124 |                 309 |
| gate_c_r0.1        |  55538 |    56988 |          -437 |            27 |                  77 |
| milp_turnover2     |  55304 |    58162 |         -4508 |          -207 |                1136 |
| milp_turnover0.5   |  55266 |    58193 |         -5302 |          -245 |                1232 |
| baseline_milp_info |  55260 |    57892 |         -5621 |          -251 |                1277 |
| milp_derate0.95    |  54974 |    58106 |         -4438 |          -537 |                1539 |
| milp_derate0.9     |  54257 |    57851 |         -5546 |         -1254 |                2273 |

Turnover penalties price the churn between successive market
re-optimizations; forecast derating plans against a conservative
renewable estimate (solver: Gurobi, 30 s limit per solve).
