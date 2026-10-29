| | Item | rc1 | tonetext-r3c-student-minilm-l6-seed2 | Difference (95% interval) | Lines |
| --- | --- | --- | --- | --- | --- |
| Voice-agent sets | Emotion macro-F1, generated calls (9 classes) | 0.5983 | 0.6771 | +0.0788 (-0.0126 to +0.116) | 885 |
| Voice-agent sets | Emotion macro-F1, real calls | 0.6643 | 0.713 | +0.0487 (-0.0779 to +0.1862) | 1496 |
| Voice-agent sets | Move macro-F1, real calls | 0.7137 | 0.6755 | -0.0382 (-0.0973 to +0.0187) | 1470 |
| Voice-agent sets | Sarcasm F1, generated calls | 0.8866 | 0.9026 | +0.016 (-0.0217 to +0.0526) | 897 |
| Voice-agent sets | Sincere turns called sarcastic, real calls | 0.00067 | 0.00067 | +0.0 (-0.00201 to +0.00201) | 1496 |
| Sarcasm sets | Sarcasm F1, held-out tests | 0.8544 | 0.8482 | -0.0062 (-0.0309 to +0.0179) | 900 |
| Sarcasm sets | Sincere lines called sarcastic, natural test | 0.01717 | 0.0103 | -0.00687 (-0.01145 to -0.00286) | 1747 |
| Sarcasm sets | Sarcastic lines read happy, Tone's tests | 0.10811 | 0.02703 | -0.08108 (-0.16216 to +0.0) | 37 |
| Tone's sets | Emotion macro-F1, main test | 0.6685 | 0.6611 | -0.0074 (-0.038 to +0.0196) | 1905 |
| Tone's sets | Emotion macro-F1, player speech | 0.7246 | 0.6843 | -0.0403 (-0.1025 to +0.0125) | 870 |
| Tone's sets | Move macro-F1 | 0.8305 | 0.818 | -0.0125 (-0.0367 to +0.0102) | 824 |
| Support-call test split | Sarcasm F1, committee labels | 0.9031 | 0.9589 | +0.0559 (+0.0438 to +0.0673) | 3752 |
| Support-call test split | Sarcasm F1 after a plain agent line | 0.9089 | 0.9675 | +0.0586 (+0.0444 to +0.0718) | 2116 |
| Support-call test split | Sarcasm F1 after news | 0.9052 | 0.9561 | +0.0509 (+0.0308 to +0.0733) | 890 |
| Support-call test split | Sarcasm F1, held-out domains | 0.9064 | 0.96 | +0.0536 (+0.0411 to +0.0669) | 3052 |
| Support-call test split | Sincere turns called sarcastic | 0.03021 | 0.03151 | +0.00129 (-0.00518 to +0.00734) | 2317 |
| Support-call test split | Sarcasm F1, as written | 0.9035 | 0.9636 | +0.0601 (+0.0494 to +0.0716) | 3006 |
| Support-call test split | Sarcasm F1 after a plain line, as written | 0.903 | 0.9637 | +0.0608 (+0.0467 to +0.0743) | 2116 |
| Support-call test split | Sarcastic turns read happy | 0.18666 | 0.02058 | -0.16608 (-0.18666 to -0.14549) | 1409 |
| Support-call test split | Emotion macro-F1 | 0.5019 | 0.626 | +0.1241 (+0.0978 to +0.2258) | 3688 |
| Support-call test split | Sarcasm F1, turn alone | 0.9215 | 0.947 | +0.0255 (+0.0163 to +0.0348) | 3752 |
| Contrast test pairs | Sarcasm F1, committee labels | 0.6832 | 0.9326 | +0.2494 (+0.209 to +0.2921) | 870 |
| Contrast test pairs | Sarcasm F1, held-out domains | 0.6983 | 0.9416 | +0.2432 (+0.1991 to +0.2906) | 644 |
| Contrast test pairs | Sincere turns called sarcastic | 0.10621 | 0.06613 | -0.04008 (-0.07214 to -0.01002) | 499 |
| Contrast test pairs | Pairs read right on both lines | 0.49041 | 0.93973 | +0.44932 (+0.39726 to +0.50411) | 365 |
| Contrast test pairs | Pairs read right on both lines, at 0.7 | 0.35616 | 0.93425 | +0.57808 (+0.52603 to +0.63014) | 365 |
| Contrast test pairs | Turns alone called sarcastic | 0.20274 | 0.00274 | -0.2 (-0.2411 to -0.16164) | 365 |
| Contrast test pairs | Sarcastic turns read happy | 0.71293 | 0.03785 | -0.67508 (-0.72871 to -0.62461) | 317 |

Cases read right: rc1 8, tonetext-r3c-student-minilm-l6-seed2 10 of 10.

| Case | Should read | rc1: emotion, sarcasm | tonetext-r3c-student-minilm-l6-seed2: emotion, sarcasm |
| --- | --- | --- | --- |
| great_late_own | sarcastic | angry 0.53, 0.65 | angry 0.9, 0.97 |
| great_late_after_status | sarcastic | happy 0.82, 0.07 ✗ | angry 0.9, 0.97 |
| wonderful_after_delay | sarcastic | angry 0.84, 0.99 | angry 0.89, 0.97 |
| wonderful_own | sincere | happy 0.95, 0.03 | happy 0.93, 0.01 |
| helpful_after_refusal | sarcastic | contemptuous 0.55, 0.9 | contemptuous 0.95, 0.98 |
| helpful_own | sincere | happy 0.94, 0.07 | happy 0.93, 0.01 |
| wonderful_after_refund | sincere | happy 0.95, 0.02 | happy 0.93, 0.01 |
| helpful_after_reset | sincere | happy 0.96, 0.02 | happy 0.93, 0.01 |
| few_days_after_delay | sarcastic | angry 0.9, 0.99 | angry 0.89, 0.97 |
| few_days_own | sarcastic | happy 0.79, 0.13 ✗ | angry 0.89, 0.97 |

Praise cases read right, by each one's call (by the chance at 0.7 in brackets), and the mean chance of sarcasm.

| Group | Should read | Lines | rc1 | tonetext-r3c-student-minilm-l6-seed2 |
| --- | --- | --- | --- | --- |
| after_good | sincere | 13 | 13 (13), 0.022 | 13 (13), 0.01 |
| alone | sincere | 8 | 8 (8), 0.046 | 8 (8), 0.015 |
| after_bad | sarcastic | 8 | 6 (6), 0.701 | 8 (8), 0.948 |
| own_words | sarcastic | 6 | 3 (2), 0.47 | 6 (6), 0.965 |
