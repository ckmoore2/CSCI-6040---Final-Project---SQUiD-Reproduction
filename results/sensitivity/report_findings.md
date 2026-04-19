# SQUiD Noise Sensitivity - Findings

## Methodology

Five metrics computed directly from Stage-4 outputs and the
built SQLite databases, bypassing SQUiD's internal
`database_evaluation.py` which fails silently due to a schema
mismatch between Stage 2 (join query) and Stage 3 (CREATE TABLE):

- **TC**  Tuple Coverage (semantic similarity of GT values to DB cells, threshold 0.7)
- **VC**  Value Coverage (exact substring presence in any DB cell)
- **CC**  Column Consistency (values landing in columns whose names match the GT attribute)
- **RI**  Referential Integrity (fraction of FK values satisfying their parent PK)
- **EX**  Executability (fraction of entries whose sql_statements run without error)

## Clean reference

- Clean SQUiD  - TC=0.856, VC=0.817, CC=0.345, RI=0.985, EX=0.993
- Clean Baseline - TC=0.157, VC=0.160, CC=0.081, RI=1.000, EX=0.161

## Per-variant table

| Variant | Method | TC | VC | CC | RI | EX |
|---|---|---|---|---|---|---|
| clean | squid | 0.856 | 0.817 | 0.345 | 0.985 | 0.993 |
| clean | baseline | 0.157 | 0.160 | 0.081 | 1.000 | 0.161 |
| bird_type1_contradictory_L1 | squid | 0.887 | 0.783 | 0.348 | 0.977 | 0.995 |
| bird_type1_contradictory_L1 | baseline | 0.170 | 0.170 | 0.087 | 1.000 | 0.172 |
| bird_type1_contradictory_L2 | squid | 0.876 | 0.772 | 0.329 | 0.982 | 0.990 |
| bird_type1_contradictory_L2 | baseline | 0.164 | 0.153 | 0.078 | 1.000 | 0.172 |
| bird_type1_contradictory_L3 | squid | 0.888 | 0.775 | 0.323 | 0.977 | 0.995 |
| bird_type1_contradictory_L3 | baseline | 0.137 | 0.139 | 0.072 | 1.000 | 0.151 |
| bird_type2_missing_L1 | squid | 0.853 | 0.758 | 0.328 | 0.975 | 1.000 |
| bird_type2_missing_L1 | baseline | 0.144 | 0.149 | 0.073 | 1.000 | 0.141 |
| bird_type2_missing_L2 | squid | 0.848 | 0.742 | 0.297 | 0.988 | 1.000 |
| bird_type2_missing_L2 | baseline | 0.133 | 0.133 | 0.064 | 1.000 | 0.130 |
| bird_type2_missing_L3 | squid | 0.786 | 0.645 | 0.252 | 0.987 | 1.000 |
| bird_type2_missing_L3 | baseline | 0.127 | 0.117 | 0.052 | 1.000 | 0.135 |
| bird_type3_duplicates_L1 | squid | 0.853 | 0.790 | 0.340 | 0.984 | 0.990 |
| bird_type3_duplicates_L1 | baseline | 0.173 | 0.172 | 0.086 | 1.000 | 0.177 |
| bird_type3_duplicates_L2 | squid | 0.888 | 0.790 | 0.338 | 0.982 | 0.995 |
| bird_type3_duplicates_L2 | baseline | 0.165 | 0.172 | 0.087 | 1.000 | 0.161 |
| bird_type3_duplicates_L3 | squid | 0.848 | 0.805 | 0.336 | 0.981 | 0.986 |
| bird_type3_duplicates_L3 | baseline | 0.160 | 0.158 | 0.078 | 1.000 | 0.156 |
| bird_type4_structural_L1 | squid | 0.884 | 0.792 | 0.334 | 0.981 | 0.995 |
| bird_type4_structural_L1 | baseline | 0.150 | 0.156 | 0.078 | 1.000 | 0.151 |
| bird_type4_structural_L2 | squid | 0.874 | 0.781 | 0.337 | 0.964 | 0.990 |
| bird_type4_structural_L2 | baseline | 0.154 | 0.160 | 0.079 | 1.000 | 0.135 |
| bird_type4_structural_L3 | squid | 0.885 | 0.783 | 0.352 | 0.980 | 0.995 |
| bird_type4_structural_L3 | baseline | 0.143 | 0.145 | 0.072 | 1.000 | 0.156 |

## Degradation per noise type

### type1_contradictory

- **squid**  L1: dTC=+0.031  L2: dTC=+0.020  L3: dTC=+0.032
- **baseline**  L1: dTC=+0.014  L2: dTC=+0.008  L3: dTC=-0.020

### type2_missing

- **squid**  L1: dTC=-0.003  L2: dTC=-0.008  L3: dTC=-0.070
- **baseline**  L1: dTC=-0.012  L2: dTC=-0.024  L3: dTC=-0.029

### type3_duplicates

- **squid**  L1: dTC=-0.003  L2: dTC=+0.032  L3: dTC=-0.008
- **baseline**  L1: dTC=+0.017  L2: dTC=+0.008  L3: dTC=+0.003

### type4_structural

- **squid**  L1: dTC=+0.028  L2: dTC=+0.018  L3: dTC=+0.029
- **baseline**  L1: dTC=-0.006  L2: dTC=-0.003  L3: dTC=-0.013

## Interpretation

**1. Largest TC drop for SQUiD.** Type 2 (missing data) at L3, dropping from 0.856 clean to 0.786 noisy. Other noise types stay within 0.03 of clean.

**2. Is the baseline more robust?** No. The baseline's metrics sit in a narrow band (TC 0.127-0.173) across all conditions, but this is because it's already operating near its performance floor. SQUiD is the more useful system even when it degrades slightly.

**3. Does degradation scale linearly with noise level?** Mostly no. For type 2 (missing), TC stays flat from L1 to L2 (0.853 to 0.848) then drops sharply at L3 (0.786), suggesting a threshold effect rather than linear decay. For contradictory, duplicate, and structural noise, there is no monotonic relationship between noise level and metric scores.

**4. Does structural noise affect RI more than value-level noise?** No. RI stays near 0.97-0.99 across every noise type and level tested. Referential integrity is surprisingly insensitive to input corruption when SQUiD generates the schema.
