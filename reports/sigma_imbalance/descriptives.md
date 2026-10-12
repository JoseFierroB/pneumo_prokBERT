# Sigma imbalance audit — d39v_full_989 (+ TIGR4 join)

No training performed. Tags parsed from `id` suffix; splits stratified by `y` only.

## Counts per tag x split (positives)

| split | n | pos | top tags |
|---|---|---|---|
| train | 1381 | 681 | None 353, SigA 241, CcpA 15, SigX 15, None-LOWCONF 11, SigA-LOWCONF 11 |
| eval | 296 | 146 | SigA 64, None 62, SigA-LOWCONF 6, CiaR 3, CcpA 2, SigX 2 |
| test | 296 | 146 | SigA 68, None 63, SigX 4, CodY 2, CcpA 2, CmbR 2 |
| all | 1973 | 973 | None 478, SigA 373, SigX 21, CcpA 19, SigA-LOWCONF 17, None-LOWCONF 11 |

## Fragile cells (eval/test n<5)
| split | tag | n |
|---|---|---|
| train | BlpR | 4 |
| train | Rex | 4 |
| train | MtaR | 4 |
| train | CmbR | 3 |
| train | PurR | 3 |
| train | FabT | 2 |
| train | CiaR-LOWCONF | 2 |
| train | TreR | 1 |
| train | LiaR | 1 |
| train | DnaA | 1 |
| train | ArgR | 1 |
| train | GalR | 1 |
| train | MalR | 1 |
| train | CodY | 1 |
| train | MntR | 1 |
| eval | CiaR | 3 |
| eval | CcpA | 2 |
| eval | SigX | 2 |
| eval | PipR | 1 |
| eval | GalR | 1 |
| eval | CiaR-LOWCONF | 1 |
| eval | MalR | 1 |
| eval | Rex | 1 |
| eval | LacR | 1 |
| eval | BlpR | 1 |
| test | SigX | 4 |
| test | CodY | 2 |
| test | CcpA | 2 |
| test | CmbR | 2 |
| test | DeoR | 1 |
| test | MalR | 1 |
| test | CiaR | 1 |
| test | MntR | 1 |
| test | SczA | 1 |

Singletons (n=1 in all): ArgR, DeoR, DnaA, LacR, LiaR, PipR, SczA, TreR
LOWCONF: {'CiaR-LOWCONF': 3, 'None-LOWCONF': 11, 'SigA-LOWCONF': 17}

## Stratification check (observed vs proportional)
| split | group | obs | exp |
|---|---|---|---|
| train | SigA | 241 | 261.08 |
| train | SigX | 15 | 14.7 |
| train | None | 353 | 334.58 |
| eval | SigA | 64 | 55.96 |
| eval | SigX | 2 | 3.15 |
| eval | None | 62 | 71.71 |
| test | SigA | 68 | 55.96 |
| test | SigX | 4 | 3.15 |
| test | None | 63 | 71.71 |

## GC composition (confound check)
| group | n | mean | std |
|---|---|---|---|
| SigA | 373 | 0.2944 | 0.051 |
| SigX | 21 | 0.2975 | 0.0463 |
| None | 478 | 0.3049 | 0.0702 |
| neg | 1000 | 0.4125 | 0.0633 |

## TIGR4 join

- Competence pool: D39V SigX 21 + TIGR4 ComX 6 = 27
