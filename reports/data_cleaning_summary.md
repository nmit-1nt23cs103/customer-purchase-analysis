### Data cleaning

| Step | Rows |
| --- | --- |
| Raw rows | 2,240 |
| Duplicate customers removed (identical except ID) | 184 |
| Missing income values imputed (education-group median) | 24 |
| Impossible ages removed (> 100 years) | 3 |
| Income outliers removed (> 165,993, Q3 + 3xIQR) | 1 |
| Marital-status oddities recoded to 'Single' (Alone / Absurd / YOLO) | 7 |
| **Clean rows** | 2,052 |

Engineered features: `age`, `tenure_days`, `total_spend`, `total_purchases`, `children`, `family_size`, `is_parent`, `campaigns_accepted`, `avg_spend_per_purchase`, `education_level`, `living_with`. Age and tenure are measured at the dataset snapshot date (2014-06-30).
