| | Item | r3 on r2's teachers | tonetext-r3c-student-minilm-l6-seed1 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.6213 | 0.6633 | +0.042 (-0.0056 to +0.0878) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.6888 | 0.6905 | +0.0018 (-0.0754 to +0.0919) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7189 | 0.6746 | -0.0443 (-0.1101 to +0.0154) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.8763 | 0.8889 | +0.0126 (-0.0215 to +0.0477) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00134 | 0.0 | -0.00134 (-0.00334 to +0.0) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8522 | 0.8554 | +0.0032 (-0.0212 to +0.0281) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01488 | 0.01317 | -0.00172 (-0.00687 to +0.00343) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.10811 | 0.08108 | -0.02703 (-0.13514 to +0.05405) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6672 | 0.6516 | -0.0156 (-0.0466 to +0.0118) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7248 | 0.7127 | -0.0121 (-0.039 to +0.0122) | 870 |
| Tone's sets | Move macro-F1 | 0.8183 | 0.8229 | +0.0046 (-0.0175 to +0.0272) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9641 | 0.9623 | -0.0017 (-0.0067 to +0.0029) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9715 | 0.9688 | -0.0027 (-0.0082 to +0.0027) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9585 | 0.9658 | +0.0073 (-0.0014 to +0.0156) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.963 | 0.9628 | -0.0001 (-0.0055 to +0.0055) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.02762 | 0.03453 | +0.00691 (+0.00216 to +0.01122) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9677 | 0.9713 | +0.0036 (-0.0005 to +0.0082) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.9696 | 0.9717 | +0.0022 (-0.0029 to +0.0071) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.01561 | 0.01207 | -0.00355 (-0.00852 to +0.00071) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.7101 | 0.6468 | -0.0633 (-0.0844 to +0.0342) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9611 | 0.9561 | -0.0051 (-0.0106 to +0.0004) | 3752 |
| Contrast test pairs | Sarcasm F1, committee labels | 0.7216 | 0.9307 | +0.2091 (+0.1717 to +0.2492) | 870 |
| Contrast test pairs | Sarcasm F1, held-out domains | 0.7419 | 0.9336 | +0.1916 (+0.1494 to +0.2356) | 644 |
| Contrast test pairs | Sincere turns called sarcastic | 0.11222 | 0.06012 | -0.0521 (-0.08216 to -0.02204) | 499 |
| Contrast test pairs | Pairs read right on both lines | 0.53973 | 0.93699 | +0.39726 (+0.34795 to +0.44932) | 365 |
| Contrast test pairs | Pairs read right on both lines, at 0.7 | 0.4411 | 0.92329 | +0.48219 (+0.4274 to +0.53425) | 365 |
| Contrast test pairs | Turns alone called sarcastic | 0.23288 | 0.00822 | -0.22466 (-0.26849 to -0.18356) | 365 |
| Contrast test pairs | Sarcastic turns read happy | 0.57729 | 0.05047 | -0.52681 (-0.58044 to -0.46688) | 317 |

Cases read right: r3 on r2's teachers 9, tonetext-r3c-student-minilm-l6-seed1 10 of 10.

| Case | Should read | r3 on r2's teachers: emotion, sarcasm | tonetext-r3c-student-minilm-l6-seed1: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.9, 0.98 | angry 0.89, 0.94 |
| great_late_after_status | sarcastic | angry 0.9, 0.98 | angry 0.9, 0.94 |
| wonderful_after_delay | sarcastic | angry 0.89, 0.98 | angry 0.89, 0.94 |
| wonderful_own | sincere | happy 0.77, 0.18 | happy 0.94, 0.01 |
| helpful_after_refusal | sarcastic | happy 0.93, 0.13 ✗ | contemptuous 0.94, 0.95 |
| helpful_own | sincere | happy 0.94, 0.02 | happy 0.94, 0.01 |
| wonderful_after_refund | sincere | happy 0.93, 0.05 | happy 0.93, 0.01 |
| helpful_after_reset | sincere | happy 0.94, 0.01 | happy 0.94, 0.01 |
| few_days_after_delay | sarcastic | angry 0.9, 0.98 | angry 0.89, 0.94 |
| few_days_own | sarcastic | angry 0.9, 0.98 | angry 0.89, 0.95 |

Praise cases read right, by each one's call (by the chance at 0.7 in brackets), and the mean chance of sarcasm.

| Group | Should read | Lines | r3 on r2's teachers | tonetext-r3c-student-minilm-l6-seed1 |
| --- | --- | --- | --- | --- |
| after_good | sincere | 13 | 13 (13), 0.022 | 13 (13), 0.01 |
| alone | sincere | 8 | 8 (8), 0.06 | 8 (8), 0.014 |
| after_bad | sarcastic | 8 | 5 (5), 0.59 | 8 (8), 0.95 |
| own_words | sarcastic | 6 | 6 (6), 0.98 | 6 (6), 0.942 |
