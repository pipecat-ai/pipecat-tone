| | Item | rc1 | tonetext-r3-student-minilm-l6-seed2 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.5983 | 0.6581 | +0.0598 (-0.0364 to +0.1072) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.6643 | 0.6924 | +0.0281 (-0.0695 to +0.1499) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7137 | 0.7109 | -0.0028 (-0.0402 to +0.0313) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.8866 | 0.8945 | +0.0079 (-0.0278 to +0.0459) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00067 | 0.00067 | +0.0 (+0.0 to +0.0) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8544 | 0.8609 | +0.0065 (-0.0168 to +0.03) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01717 | 0.01374 | -0.00343 (-0.00744 to +0.00057) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.10811 | 0.08108 | -0.02703 (-0.08108 to +0.0) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6685 | 0.6906 | +0.0221 (-0.0013 to +0.0467) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7246 | 0.6921 | -0.0325 (-0.0822 to +0.0147) | 870 |
| Tone's sets | Move macro-F1 | 0.8305 | 0.8344 | +0.0039 (-0.0162 to +0.0243) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9031 | 0.9663 | +0.0632 (+0.0526 to +0.0743) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9089 | 0.9721 | +0.0632 (+0.0496 to +0.0764) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9052 | 0.9668 | +0.0616 (+0.0426 to +0.0819) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.9064 | 0.9678 | +0.0614 (+0.049 to +0.0741) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.03021 | 0.02935 | -0.00086 (-0.00734 to +0.00518) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9035 | 0.9712 | +0.0677 (+0.0576 to +0.0786) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.903 | 0.9711 | +0.0681 (+0.0544 to +0.0818) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.18666 | 0.01419 | -0.17246 (-0.19233 to -0.15259) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.5019 | 0.7371 | +0.2352 (+0.2014 to +0.2667) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9215 | 0.9628 | +0.0414 (+0.0323 to +0.0508) | 3752 |

Cases read right: rc1 8, tonetext-r3-student-minilm-l6-seed2 8 of 10.

| Case | Should read | rc1: emotion, sarcasm | tonetext-r3-student-minilm-l6-seed2: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.53, 0.65 | angry 0.91, 0.98 |
| great_late_after_status | sarcastic | happy 0.82, 0.07 ✗ | angry 0.9, 0.98 |
| wonderful_after_delay | sarcastic | angry 0.84, 0.99 | angry 0.89, 0.98 |
| wonderful_own | sincere | happy 0.95, 0.03 | angry 0.57, 0.69 ✗ |
| helpful_after_refusal | sarcastic | contemptuous 0.55, 0.9 | happy 0.87, 0.07 ✗ |
| helpful_own | sincere | happy 0.94, 0.07 | happy 0.92, 0.02 |
| wonderful_after_refund | sincere | happy 0.95, 0.02 | happy 0.9, 0.06 |
| helpful_after_reset | sincere | happy 0.96, 0.02 | happy 0.93, 0.01 |
| few_days_after_delay | sarcastic | angry 0.9, 0.99 | angry 0.9, 0.98 |
| few_days_own | sarcastic | happy 0.79, 0.13 ✗ | angry 0.9, 0.98 |
