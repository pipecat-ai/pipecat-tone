| | Item | rc1 | tonetext-r3c-student-minilm-l6-seed1 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.5983 | 0.6633 | +0.065 (-0.0254 to +0.1013) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.6643 | 0.6905 | +0.0262 (-0.0701 to +0.1295) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7137 | 0.6746 | -0.0391 (-0.1015 to +0.0203) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.8866 | 0.8889 | +0.0023 (-0.0417 to +0.0471) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00067 | 0.0 | -0.00067 (-0.00201 to +0.0) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8544 | 0.8554 | +0.001 (-0.0218 to +0.0244) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01717 | 0.01317 | -0.00401 (-0.00916 to +0.00114) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.10811 | 0.08108 | -0.02703 (-0.08108 to +0.0) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6685 | 0.6516 | -0.0169 (-0.0478 to +0.009) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7246 | 0.7127 | -0.0119 (-0.0718 to +0.029) | 870 |
| Tone's sets | Move macro-F1 | 0.8305 | 0.8229 | -0.0076 (-0.0296 to +0.015) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9031 | 0.9623 | +0.0593 (+0.0475 to +0.0708) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9089 | 0.9688 | +0.0599 (+0.0456 to +0.0736) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9052 | 0.9658 | +0.0605 (+0.0412 to +0.0803) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.9064 | 0.9628 | +0.0564 (+0.0446 to +0.0695) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.03021 | 0.03453 | +0.00432 (-0.00259 to +0.01122) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9035 | 0.9713 | +0.0678 (+0.0572 to +0.0794) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.903 | 0.9717 | +0.0688 (+0.0545 to +0.0823) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.18666 | 0.01207 | -0.17459 (-0.19517 to -0.1533) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.5019 | 0.6468 | +0.1449 (+0.1205 to +0.246) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9215 | 0.9561 | +0.0346 (+0.0256 to +0.0442) | 3752 |
| Contrast test pairs | Sarcasm F1, committee labels | 0.6832 | 0.9307 | +0.2474 (+0.2076 to +0.2914) | 870 |
| Contrast test pairs | Sarcasm F1, held-out domains | 0.6983 | 0.9336 | +0.2352 (+0.1909 to +0.2822) | 644 |
| Contrast test pairs | Sincere turns called sarcastic | 0.10621 | 0.06012 | -0.04609 (-0.07816 to -0.01403) | 499 |
| Contrast test pairs | Pairs read right on both lines | 0.49041 | 0.93699 | +0.44658 (+0.39726 to +0.49863) | 365 |
| Contrast test pairs | Pairs read right on both lines, at 0.7 | 0.35616 | 0.92329 | +0.56712 (+0.51507 to +0.61918) | 365 |
| Contrast test pairs | Turns alone called sarcastic | 0.20274 | 0.00822 | -0.19452 (-0.23562 to -0.15616) | 365 |
| Contrast test pairs | Sarcastic turns read happy | 0.71293 | 0.05047 | -0.66246 (-0.71609 to -0.60568) | 317 |

Cases read right: rc1 8, tonetext-r3c-student-minilm-l6-seed1 10 of 10.

| Case | Should read | rc1: emotion, sarcasm | tonetext-r3c-student-minilm-l6-seed1: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.53, 0.65 | angry 0.89, 0.94 |
| great_late_after_status | sarcastic | happy 0.82, 0.07 ✗ | angry 0.9, 0.94 |
| wonderful_after_delay | sarcastic | angry 0.84, 0.99 | angry 0.89, 0.94 |
| wonderful_own | sincere | happy 0.95, 0.03 | happy 0.94, 0.01 |
| helpful_after_refusal | sarcastic | contemptuous 0.55, 0.9 | contemptuous 0.94, 0.95 |
| helpful_own | sincere | happy 0.94, 0.07 | happy 0.94, 0.01 |
| wonderful_after_refund | sincere | happy 0.95, 0.02 | happy 0.93, 0.01 |
| helpful_after_reset | sincere | happy 0.96, 0.02 | happy 0.94, 0.01 |
| few_days_after_delay | sarcastic | angry 0.9, 0.99 | angry 0.89, 0.94 |
| few_days_own | sarcastic | happy 0.79, 0.13 ✗ | angry 0.89, 0.95 |

Praise cases read right, by each one's call (by the chance at 0.7 in brackets), and the mean chance of sarcasm.

| Group | Should read | Lines | rc1 | tonetext-r3c-student-minilm-l6-seed1 |
| --- | --- | --- | --- | --- |
| after_good | sincere | 13 | 13 (13), 0.022 | 13 (13), 0.01 |
| alone | sincere | 8 | 8 (8), 0.046 | 8 (8), 0.014 |
| after_bad | sarcastic | 8 | 6 (6), 0.701 | 8 (8), 0.95 |
| own_words | sarcastic | 6 | 3 (2), 0.47 | 6 (6), 0.942 |
