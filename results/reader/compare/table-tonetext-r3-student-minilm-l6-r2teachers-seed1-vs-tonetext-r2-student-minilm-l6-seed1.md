| | Item | rc1 | tonetext-r3-student-minilm-l6-r2teachers-seed1 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.5983 | 0.6213 | +0.023 (-0.0741 to +0.078) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.6643 | 0.6888 | +0.0244 (-0.065 to +0.1154) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7137 | 0.7189 | +0.0052 (-0.0266 to +0.0437) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.8866 | 0.8763 | -0.0103 (-0.0534 to +0.0304) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00067 | 0.00134 | +0.00067 (+0.0 to +0.00201) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8544 | 0.8522 | -0.0022 (-0.0229 to +0.0174) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01717 | 0.01488 | -0.00229 (-0.0063 to +0.00114) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.10811 | 0.10811 | +0.0 (-0.08108 to +0.08108) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6685 | 0.6672 | -0.0014 (-0.0229 to +0.0214) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7246 | 0.7248 | +0.0002 (-0.0581 to +0.0388) | 870 |
| Tone's sets | Move macro-F1 | 0.8305 | 0.8183 | -0.0123 (-0.0338 to +0.0107) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9031 | 0.9641 | +0.061 (+0.0506 to +0.072) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9089 | 0.9715 | +0.0626 (+0.0483 to +0.0765) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9052 | 0.9585 | +0.0532 (+0.0345 to +0.0742) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.9064 | 0.963 | +0.0566 (+0.0446 to +0.0692) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.03021 | 0.02762 | -0.00259 (-0.00863 to +0.00345) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9035 | 0.9677 | +0.0642 (+0.0535 to +0.0751) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.903 | 0.9696 | +0.0666 (+0.0521 to +0.0803) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.18666 | 0.01561 | -0.17104 (-0.19092 to -0.15046) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.5019 | 0.7101 | +0.2082 (+0.1773 to +0.2348) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9215 | 0.9611 | +0.0397 (+0.0308 to +0.049) | 3752 |

Cases read right: rc1 8, tonetext-r3-student-minilm-l6-r2teachers-seed1 9 of 10.

| Case | Should read | rc1: emotion, sarcasm | tonetext-r3-student-minilm-l6-r2teachers-seed1: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.53, 0.65 | angry 0.9, 0.98 |
| great_late_after_status | sarcastic | happy 0.82, 0.07 ✗ | angry 0.9, 0.98 |
| wonderful_after_delay | sarcastic | angry 0.84, 0.99 | angry 0.89, 0.98 |
| wonderful_own | sincere | happy 0.95, 0.03 | happy 0.77, 0.18 |
| helpful_after_refusal | sarcastic | contemptuous 0.55, 0.9 | happy 0.93, 0.13 ✗ |
| helpful_own | sincere | happy 0.94, 0.07 | happy 0.94, 0.02 |
| wonderful_after_refund | sincere | happy 0.95, 0.02 | happy 0.93, 0.05 |
| helpful_after_reset | sincere | happy 0.96, 0.02 | happy 0.94, 0.01 |
| few_days_after_delay | sarcastic | angry 0.9, 0.99 | angry 0.9, 0.98 |
| few_days_own | sarcastic | happy 0.79, 0.13 ✗ | angry 0.9, 0.98 |
