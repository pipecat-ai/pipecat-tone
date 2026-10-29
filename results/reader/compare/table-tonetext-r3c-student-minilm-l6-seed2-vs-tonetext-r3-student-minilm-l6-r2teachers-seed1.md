| | Item | r3 on r2's teachers | tonetext-r3c-student-minilm-l6-seed2 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.6213 | 0.6771 | +0.0558 (+0.0007 to +0.1068) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.6888 | 0.713 | +0.0242 (-0.0786 to +0.1457) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7189 | 0.6755 | -0.0434 (-0.1062 to +0.0148) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.8763 | 0.9026 | +0.0263 (-0.0059 to +0.0633) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00134 | 0.00067 | -0.00067 (-0.00267 to +0.00134) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8522 | 0.8482 | -0.004 (-0.0252 to +0.0179) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01488 | 0.0103 | -0.00458 (-0.00916 to -0.00057) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.10811 | 0.02703 | -0.08108 (-0.18919 to +0.0) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6672 | 0.6611 | -0.006 (-0.0382 to +0.0222) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7248 | 0.6843 | -0.0405 (-0.1032 to +0.0076) | 870 |
| Tone's sets | Move macro-F1 | 0.8183 | 0.818 | -0.0003 (-0.0204 to +0.0207) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9641 | 0.9589 | -0.0051 (-0.0103 to -0.0002) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9715 | 0.9675 | -0.004 (-0.0096 to +0.0019) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9585 | 0.9561 | -0.0024 (-0.0135 to +0.0084) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.963 | 0.96 | -0.003 (-0.0092 to +0.0029) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.02762 | 0.03151 | +0.00388 (-0.00086 to +0.00863) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9677 | 0.9636 | -0.0041 (-0.0093 to +0.0011) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.9696 | 0.9637 | -0.0058 (-0.0118 to +0.0) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.01561 | 0.02058 | +0.00497 (-0.00213 to +0.01207) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.7101 | 0.626 | -0.0841 (-0.103 to +0.0105) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9611 | 0.947 | -0.0141 (-0.0204 to -0.0082) | 3752 |
| Contrast test pairs | Sarcasm F1, committee labels | 0.7216 | 0.9326 | +0.2111 (+0.1721 to +0.2509) | 870 |
| Contrast test pairs | Sarcasm F1, held-out domains | 0.7419 | 0.9416 | +0.1997 (+0.1568 to +0.2425) | 644 |
| Contrast test pairs | Sincere turns called sarcastic | 0.11222 | 0.06613 | -0.04609 (-0.07415 to -0.01603) | 499 |
| Contrast test pairs | Pairs read right on both lines | 0.53973 | 0.93973 | +0.4 (+0.35068 to +0.45479) | 365 |
| Contrast test pairs | Pairs read right on both lines, at 0.7 | 0.4411 | 0.93425 | +0.49315 (+0.43836 to +0.54521) | 365 |
| Contrast test pairs | Turns alone called sarcastic | 0.23288 | 0.00274 | -0.23014 (-0.27671 to -0.18904) | 365 |
| Contrast test pairs | Sarcastic turns read happy | 0.57729 | 0.03785 | -0.53943 (-0.59621 to -0.4795) | 317 |

Cases read right: r3 on r2's teachers 9, tonetext-r3c-student-minilm-l6-seed2 10 of 10.

| Case | Should read | r3 on r2's teachers: emotion, sarcasm | tonetext-r3c-student-minilm-l6-seed2: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.9, 0.98 | angry 0.9, 0.97 |
| great_late_after_status | sarcastic | angry 0.9, 0.98 | angry 0.9, 0.97 |
| wonderful_after_delay | sarcastic | angry 0.89, 0.98 | angry 0.89, 0.97 |
| wonderful_own | sincere | happy 0.77, 0.18 | happy 0.93, 0.01 |
| helpful_after_refusal | sarcastic | happy 0.93, 0.13 ✗ | contemptuous 0.95, 0.98 |
| helpful_own | sincere | happy 0.94, 0.02 | happy 0.93, 0.01 |
| wonderful_after_refund | sincere | happy 0.93, 0.05 | happy 0.93, 0.01 |
| helpful_after_reset | sincere | happy 0.94, 0.01 | happy 0.93, 0.01 |
| few_days_after_delay | sarcastic | angry 0.9, 0.98 | angry 0.89, 0.97 |
| few_days_own | sarcastic | angry 0.9, 0.98 | angry 0.89, 0.97 |

Praise cases read right, by each one's call (by the chance at 0.7 in brackets), and the mean chance of sarcasm.

| Group | Should read | Lines | r3 on r2's teachers | tonetext-r3c-student-minilm-l6-seed2 |
| --- | --- | --- | --- | --- |
| after_good | sincere | 13 | 13 (13), 0.022 | 13 (13), 0.01 |
| alone | sincere | 8 | 8 (8), 0.06 | 8 (8), 0.015 |
| after_bad | sarcastic | 8 | 5 (5), 0.59 | 8 (8), 0.948 |
| own_words | sarcastic | 6 | 6 (6), 0.98 | 6 (6), 0.965 |
