# SigA-family vs SigX (ComX) separation — diagnostic (no training)

P(promoter) from promoter-vs-nonpromoter models used as ranking score, labels SigA-family=0 / SigX=1.
d39v-test is honest (unseen); d39v-all is train_overlap; tigr4 clean except dedup-trained models.

| model | set | n0 | n1 | AUC | AP | MCC@0.5 | mean_neg | mean_pos | leakage |
|---|---|---|---|---|---|---|---|---|---|
| mini-223k-replica | d39v-test | 68 | 4 | 0.0184 | 0.0355 | +0.0000 | 0.996 | 0.966 | clean |
| mini-223k-replica | d39v-all | 390 | 21 | 0.1026 | 0.0290 | +0.0115 | 0.993 | 0.966 | train_overlap |
| mini-223k-replica | tigr4 | 278 | 6 | 0.1984 | 0.0145 | +0.0124 | 0.989 | 0.900 | clean-except-dedup-trained |
| mini-223k-replica | combined | 668 | 27 | 0.1267 | 0.0218 | +0.0132 | 0.992 | 0.951 | mixed |
| mini-d39v-tigr4 | d39v-test | 68 | 4 | 0.4375 | 0.0752 | +0.0000 | 0.970 | 0.926 | clean |
| mini-d39v-tigr4 | d39v-all | 390 | 21 | 0.5132 | 0.0642 | +0.0419 | 0.941 | 0.968 | train_overlap |
| mini-d39v-tigr4 | tigr4 | 278 | 6 | 0.6211 | 0.0573 | -0.1048 | 0.942 | 0.833 | clean-except-dedup-trained |
| mini-d39v-tigr4 | combined | 668 | 27 | 0.5376 | 0.0542 | -0.0044 | 0.941 | 0.938 | mixed |
| mini-223k-d39v-tigr4 | d39v-test | 68 | 4 | 0.1140 | 0.0380 | +0.0000 | 0.995 | 0.984 | clean |
| mini-223k-d39v-tigr4 | d39v-all | 390 | 21 | 0.1468 | 0.0296 | +0.0115 | 0.991 | 0.950 | train_overlap |
| mini-223k-d39v-tigr4 | tigr4 | 278 | 6 | 0.2968 | 0.0167 | +0.0087 | 0.984 | 0.982 | clean-except-dedup-trained |
| mini-223k-d39v-tigr4 | combined | 668 | 27 | 0.1854 | 0.0229 | +0.0108 | 0.988 | 0.957 | mixed |
| promoter-d39v-tigr4 | d39v-test | 68 | 4 | 0.2022 | 0.0417 | -0.4893 | 0.996 | 0.768 | clean |
| promoter-d39v-tigr4 | d39v-all | 390 | 21 | 0.1915 | 0.0310 | -0.3774 | 0.992 | 0.795 | train_overlap |
| promoter-d39v-tigr4 | tigr4 | 278 | 6 | 0.3525 | 0.0181 | +0.0176 | 0.979 | 0.971 | clean-except-dedup-trained |
| promoter-d39v-tigr4 | combined | 668 | 27 | 0.2287 | 0.0241 | -0.2404 | 0.986 | 0.834 | mixed |
| promoter-baseline | d39v-test | 68 | 4 | 0.1985 | 0.0419 | -0.4893 | 0.999 | 0.808 | clean |
| promoter-baseline | d39v-all | 390 | 21 | 0.1589 | 0.0298 | -0.3147 | 0.996 | 0.865 | train_overlap |
| promoter-baseline | tigr4 | 278 | 6 | 0.3171 | 0.0170 | +0.0087 | 0.993 | 0.992 | clean-except-dedup-trained |
| promoter-baseline | combined | 668 | 27 | 0.1942 | 0.0231 | -0.2472 | 0.995 | 0.893 | mixed |
