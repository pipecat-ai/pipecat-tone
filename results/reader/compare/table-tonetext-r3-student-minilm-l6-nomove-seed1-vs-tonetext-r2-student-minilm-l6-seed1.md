| | Item | tonetext-r2-student-minilm-l6-seed1 | tonetext-r3-student-minilm-l6-nomove-seed1 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.5983 | 0.6633 | +0.0651 (-0.0335 to +0.1105) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.6643 | 0.6091 | -0.0552 (-0.1759 to +0.0893) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7137 | 0.08 | -0.6337 (-0.6962 to -0.5483) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.8866 | 0.8788 | -0.0078 (-0.0492 to +0.0318) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00067 | 0.00201 | +0.00134 (+0.0 to +0.00334) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8544 | 0.8303 | -0.0242 (-0.049 to -0.0) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01717 | 0.02061 | +0.00343 (-0.00172 to +0.00859) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.10811 | 0.05405 | -0.05405 (-0.13514 to +0.0) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6685 | 0.6619 | -0.0066 (-0.0294 to +0.0168) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7246 | 0.6747 | -0.0499 (-0.1068 to -0.0015) | 870 |
| Tone's sets | Move macro-F1 | 0.8305 | 0.0813 | -0.7492 (-0.7795 to -0.7159) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9031 | 0.966 | +0.063 (+0.0518 to +0.0745) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9089 | 0.9741 | +0.0652 (+0.051 to +0.0784) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9052 | 0.9621 | +0.0568 (+0.0377 to +0.0781) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.9064 | 0.9678 | +0.0614 (+0.0491 to +0.0749) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.03021 | 0.03064 | +0.00043 (-0.00604 to +0.00691) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9035 | 0.9739 | +0.0704 (+0.0603 to +0.081) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.903 | 0.975 | +0.072 (+0.0594 to +0.0852) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.18666 | 0.01065 | -0.17601 (-0.19659 to -0.15543) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.5019 | 0.7087 | +0.2068 (+0.1663 to +0.2405) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9215 | 0.9627 | +0.0412 (+0.0317 to +0.0505) | 3752 |

Cases read right: tonetext-r2-student-minilm-l6-seed1 8, tonetext-r3-student-minilm-l6-nomove-seed1 8 of 10.

| Case | Should read | tonetext-r2-student-minilm-l6-seed1: emotion, sarcasm | tonetext-r3-student-minilm-l6-nomove-seed1: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.53, 0.65 | angry 0.91, 0.98 |
| great_late_after_status | sarcastic | happy 0.82, 0.07 ✗ | angry 0.91, 0.98 |
| wonderful_after_delay | sarcastic | angry 0.84, 0.99 | angry 0.9, 0.99 |
| wonderful_own | sincere | happy 0.95, 0.03 | angry 0.41, 0.51 ✗ |
| helpful_after_refusal | sarcastic | contemptuous 0.55, 0.9 | happy 0.87, 0.18 ✗ |
| helpful_own | sincere | happy 0.94, 0.07 | happy 0.94, 0.02 |
| wonderful_after_refund | sincere | happy 0.95, 0.02 | happy 0.89, 0.05 |
| helpful_after_reset | sincere | happy 0.96, 0.02 | happy 0.94, 0.02 |
| few_days_after_delay | sarcastic | angry 0.9, 0.99 | angry 0.91, 0.98 |
| few_days_own | sarcastic | happy 0.79, 0.13 ✗ | angry 0.92, 0.98 |
