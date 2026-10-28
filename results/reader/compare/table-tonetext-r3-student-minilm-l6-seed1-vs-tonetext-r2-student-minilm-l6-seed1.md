| | Item | rc1 | tonetext-r3-student-minilm-l6-seed1 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.5983 | 0.694 | +0.0957 (+0.0006 to +0.1382) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.6643 | 0.665 | +0.0007 (-0.1076 to +0.1063) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7137 | 0.7397 | +0.026 (-0.0337 to +0.0979) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.8866 | 0.899 | +0.0124 (-0.0271 to +0.0521) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00067 | 0.00267 | +0.00201 (+0.0 to +0.00468) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8544 | 0.8447 | -0.0097 (-0.0331 to +0.0145) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01717 | 0.01774 | +0.00057 (-0.00343 to +0.00515) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.10811 | 0.05405 | -0.05405 (-0.13514 to +0.0) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6685 | 0.6501 | -0.0184 (-0.0473 to +0.0086) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7246 | 0.7062 | -0.0184 (-0.0729 to +0.0267) | 870 |
| Tone's sets | Move macro-F1 | 0.8305 | 0.829 | -0.0015 (-0.0199 to +0.0164) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9031 | 0.9646 | +0.0615 (+0.0507 to +0.0725) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9089 | 0.9726 | +0.0637 (+0.0504 to +0.077) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9052 | 0.9632 | +0.058 (+0.0384 to +0.079) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.9064 | 0.9665 | +0.0601 (+0.049 to +0.0729) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.03021 | 0.03064 | +0.00043 (-0.00604 to +0.00647) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9035 | 0.9698 | +0.0663 (+0.0558 to +0.0768) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.903 | 0.9716 | +0.0687 (+0.0556 to +0.0818) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.18666 | 0.01348 | -0.17317 (-0.19304 to -0.1533) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.5019 | 0.731 | +0.2292 (+0.1928 to +0.2612) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9215 | 0.9613 | +0.0399 (+0.0304 to +0.0493) | 3752 |

Cases read right: rc1 8, tonetext-r3-student-minilm-l6-seed1 8 of 10.

| Case | Should read | rc1: emotion, sarcasm | tonetext-r3-student-minilm-l6-seed1: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.53, 0.65 | angry 0.9, 0.99 |
| great_late_after_status | sarcastic | happy 0.82, 0.07 ✗ | angry 0.9, 0.99 |
| wonderful_after_delay | sarcastic | angry 0.84, 0.99 | angry 0.9, 0.99 |
| wonderful_own | sincere | happy 0.95, 0.03 | happy 0.42, 0.62 ✗ |
| helpful_after_refusal | sarcastic | contemptuous 0.55, 0.9 | happy 0.61, 0.55 |
| helpful_own | sincere | happy 0.94, 0.07 | happy 0.94, 0.02 |
| wonderful_after_refund | sincere | happy 0.95, 0.02 | happy 0.73, 0.42 ✗ |
| helpful_after_reset | sincere | happy 0.96, 0.02 | happy 0.94, 0.01 |
| few_days_after_delay | sarcastic | angry 0.9, 0.99 | angry 0.91, 0.99 |
| few_days_own | sarcastic | happy 0.79, 0.13 ✗ | angry 0.91, 0.99 |
