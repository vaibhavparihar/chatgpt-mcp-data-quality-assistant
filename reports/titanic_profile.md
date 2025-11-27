# Dataset profile: `titanic`

## Overview

- File: `titanic.csv`
- Rows: **887**
- Columns: **8**

## Column summary

| Column | Dtype | Non-null | Missing % | Unique |
|--------|-------|----------|-----------|--------|
| Survived | int64 | 887 | 0.0% | 2 |
| Pclass | int64 | 887 | 0.0% | 3 |
| Name | object | 887 | 0.0% | 887 |
| Sex | object | 887 | 0.0% | 2 |
| Age | float64 | 887 | 0.0% | 89 |
| Siblings/Spouses Aboard | int64 | 887 | 0.0% | 7 |
| Parents/Children Aboard | int64 | 887 | 0.0% | 7 |
| Fare | float64 | 887 | 0.0% | 248 |

## Numeric summary (`pandas.describe()`)

```text
                         count    mean     std   min     25%     50%     75%      max
Survived                 887.0   0.386   0.487  0.00   0.000   0.000   1.000    1.000
Pclass                   887.0   2.306   0.837  1.00   2.000   3.000   3.000    3.000
Age                      887.0  29.471  14.122  0.42  20.250  28.000  38.000   80.000
Siblings/Spouses Aboard  887.0   0.525   1.105  0.00   0.000   0.000   1.000    8.000
Parents/Children Aboard  887.0   0.383   0.807  0.00   0.000   0.000   0.000    6.000
Fare                     887.0  32.305  49.782  0.00   7.925  14.454  31.138  512.329
```
## Correlations (numeric columns)

```text
                         Survived  Pclass    Age  Siblings/Spouses Aboard  Parents/Children Aboard   Fare
Survived                    1.000  -0.337 -0.060                   -0.037                    0.080  0.256
Pclass                     -0.337   1.000 -0.391                    0.085                    0.020 -0.549
Age                        -0.060  -0.391  1.000                   -0.298                   -0.194  0.112
Siblings/Spouses Aboard    -0.037   0.085 -0.298                    1.000                    0.414  0.159
Parents/Children Aboard     0.080   0.020 -0.194                    0.414                    1.000  0.215
Fare                        0.256  -0.549  0.112                    0.159                    0.215  1.000
```