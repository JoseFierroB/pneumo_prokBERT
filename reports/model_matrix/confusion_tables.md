# Confusion matrices @0.5

Rows = actual (0 neg, 1 pos), columns = predicted. From `metrics.json`.

## mini-223k-replica

### paper-test (n=22310, ACC 0.7961, MCC +0.5941)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 9222 | 1862 |
| 1 | 2686 | 8540 |

### ecoli-sigma70 (n=1864, ACC 0.8492, MCC +0.7052)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 801 | 199 |
| 1 | 82 | 782 |

### multispecies (n=22582, ACC 0.7934, MCC +0.5882)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 9350 | 1938 |
| 1 | 2728 | 8566 |

### d39v-tigr4-dedup (n=3135, ACC 0.8785, MCC +0.7589)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 1481 | 255 |
| 1 | 126 | 1273 |

## mini-d39v-tigr4

### paper-test (n=22310, ACC 0.5470, MCC +0.0976)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 7256 | 3828 |
| 1 | 6278 | 4948 |

### ecoli-sigma70 (n=1864, ACC 0.4533, MCC -0.0740)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 307 | 693 |
| 1 | 326 | 538 |

### multispecies (n=22582, ACC 0.5498, MCC +0.1017)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 7348 | 3940 |
| 1 | 6227 | 5067 |

### d39v-tigr4-dedup (n=3135, ACC 0.9282, MCC +0.8578)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 1703 | 33 |
| 1 | 192 | 1207 |

## mini-223k-d39v-tigr4

### paper-test (n=22310, ACC 0.7959, MCC +0.5922)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 8992 | 2092 |
| 1 | 2461 | 8765 |

### ecoli-sigma70 (n=1864, ACC 0.8686, MCC +0.7364)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 866 | 134 |
| 1 | 111 | 753 |

### multispecies (n=22582, ACC 0.7909, MCC +0.5820)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 9081 | 2207 |
| 1 | 2515 | 8779 |

### d39v-tigr4-dedup (n=3135, ACC 0.8957, MCC +0.7924)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 1519 | 217 |
| 1 | 110 | 1289 |

## promoter-d39v-tigr4

### paper-test (n=22310, ACC 0.8889, MCC +0.7928)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 10913 | 171 |
| 1 | 2307 | 8919 |

### ecoli-sigma70 (n=1864, ACC 0.8589, MCC +0.7217)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 939 | 61 |
| 1 | 202 | 662 |

### multispecies (n=22582, ACC 0.7910, MCC +0.6022)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 10377 | 911 |
| 1 | 3808 | 7486 |

### d39v-tigr4-dedup (n=3135, ACC 0.9301, MCC +0.8595)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 1677 | 59 |
| 1 | 160 | 1239 |

## promoter-baseline

### paper-test (n=22310, ACC 0.9239, MCC +0.8480)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 10119 | 965 |
| 1 | 732 | 10494 |

### ecoli-sigma70 (n=1864, ACC 0.8509, MCC +0.7078)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 805 | 195 |
| 1 | 83 | 781 |

### multispecies (n=22582, ACC 0.7950, MCC +0.5908)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 8692 | 2596 |
| 1 | 2033 | 9261 |

### d39v-tigr4-dedup (n=3135, ACC 0.8510, MCC +0.7102)

| actual \ pred | 0 | 1 |
|---|---|---|
| 0 | 1392 | 344 |
| 1 | 123 | 1276 |
