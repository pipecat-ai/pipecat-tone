| | Item | tonetext-r3-student-minilm-l6-seed1 | tonetext-r3-student-minilm-l6-nomove-seed1 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.694 | 0.6633 | -0.0307 (-0.0782 to +0.0182) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.665 | 0.6091 | -0.0559 (-0.1291 to +0.0518) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7397 | 0.08 | -0.6597 (-0.7277 to -0.5639) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.899 | 0.8788 | -0.0202 (-0.0568 to +0.0135) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00267 | 0.00201 | -0.00067 (-0.00201 to +0.0) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8447 | 0.8303 | -0.0144 (-0.0366 to +0.0062) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01774 | 0.02061 | +0.00286 (-0.00172 to +0.00687) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.05405 | 0.05405 | +0.0 (+0.0 to +0.0) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6501 | 0.6619 | +0.0118 (-0.0131 to +0.0402) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7062 | 0.6747 | -0.0314 (-0.0724 to +0.0021) | 870 |
| Tone's sets | Move macro-F1 | 0.829 | 0.0813 | -0.7477 (-0.7771 to -0.7145) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9646 | 0.966 | +0.0014 (-0.0029 to +0.0061) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9726 | 0.9741 | +0.0015 (-0.0022 to +0.0056) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9632 | 0.9621 | -0.0011 (-0.0089 to +0.0073) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.9665 | 0.9678 | +0.0013 (-0.0035 to +0.0063) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.03064 | 0.03064 | +0.0 (-0.00388 to +0.00388) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9698 | 0.9739 | +0.0041 (+0.0006 to +0.0076) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.9716 | 0.975 | +0.0033 (-0.0001 to +0.0071) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.01348 | 0.01065 | -0.00284 (-0.00781 to +0.00213) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.731 | 0.7087 | -0.0224 (-0.0532 to +0.0064) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9613 | 0.9627 | +0.0013 (-0.0032 to +0.0057) | 3752 |

Cases read right: tonetext-r3-student-minilm-l6-seed1 8, tonetext-r3-student-minilm-l6-nomove-seed1 8 of 10.

| Case | Should read | tonetext-r3-student-minilm-l6-seed1: emotion, sarcasm | tonetext-r3-student-minilm-l6-nomove-seed1: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.9, 0.99 | angry 0.91, 0.98 |
| great_late_after_status | sarcastic | angry 0.9, 0.99 | angry 0.91, 0.98 |
| wonderful_after_delay | sarcastic | angry 0.9, 0.99 | angry 0.9, 0.99 |
| wonderful_own | sincere | happy 0.42, 0.62 ✗ | angry 0.41, 0.51 ✗ |
| helpful_after_refusal | sarcastic | happy 0.61, 0.55 | happy 0.87, 0.18 ✗ |
| helpful_own | sincere | happy 0.94, 0.02 | happy 0.94, 0.02 |
| wonderful_after_refund | sincere | happy 0.73, 0.42 ✗ | happy 0.89, 0.05 |
| helpful_after_reset | sincere | happy 0.94, 0.01 | happy 0.94, 0.02 |
| few_days_after_delay | sarcastic | angry 0.91, 0.99 | angry 0.91, 0.98 |
| few_days_own | sarcastic | angry 0.91, 0.99 | angry 0.92, 0.98 |
