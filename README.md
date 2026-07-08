# Los Angeles Crime Data Analysis (2020–2024)

Exploratory data analysis and predictive modeling on the City of Los Angeles crime incident dataset, covering reported crimes from January 2020 through December 2024. The project combines descriptive statistics, visual analytics, and a supervised machine learning model to characterize victim demographics, weapon usage, and case-resolution patterns, and to evaluate the extent to which case status can be predicted from a small set of structured features.

## Table of Contents

- [Dataset](#dataset)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Predictive Modeling](#predictive-modeling)
- [Results and Discussion](#results-and-discussion)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [Requirements and Usage](#requirements-and-usage)
- [License](#license)

## Dataset

The source file, `Crime_Data_from_2020_to_2024.csv`, was obtained from the [City of Los Angeles – Crime Data from 2020 to Present](https://catalog.data.gov/dataset/crime-data-from-2020-to-present?from_hint=eyJzb3J0IjoicG9wdWxhcml0eSJ9) dataset, published via data.gov. It contains **1,004,894 records and 28 columns**, each representing a crime incident reported to the Los Angeles Police Department. The data used in this analysis spans **January 1, 2020 to December 31, 2023**. Key fields include the report and occurrence dates, time of occurrence, reporting area, crime code and description, victim age/sex/descent, premise type, weapon used, investigative status, and geographic coordinates (latitude/longitude).

At ingestion, the raw dataset exhibits substantial missingness in several fields relevant to this analysis:

| Field | Non-null records | Missing (%) |
|---|---|---|
| `Vict Sex` | 860,263 | ~14.4% |
| `Vict Descent` | 860,251 | ~14.4% |
| `Weapon Used Cd` / `Weapon Desc` | 327,216 | ~67.4% |
| `Mocodes` | 853,296 | ~15.1% |
| `Cross Street` | 154,228 | ~84.7% |

The high missingness in `Weapon Used Cd` reflects the fact that a majority of reported offenses (e.g., theft, fraud, identity theft) do not involve a weapon, rather than a systematic data-quality issue.

## Data Cleaning and Preprocessing

Preprocessing is implemented in `los_angeles_data_analysis.py` (`clear_data` function) and applied consistently across both the descriptive and predictive stages of the project:

1. **Column removal** — `DR_NO`, `Date Rptd`, `DATE OCC`, `Crm Cd 1–4`, and `Cross Street` were dropped, as they are either non-informative identifiers or redundant with `Crm Cd`.
2. **Row-wise filtering** — Records with missing `Vict Sex`, `Vict Descent`, `Mocodes`, `Premis Cd`, `Premis Desc`, or `Status` were removed via listwise deletion to retain only fully specified victim/case records for demographic analysis.
3. **Duplicate removal** — Exact duplicate rows were dropped.
4. **Time formatting** — The `TIME OCC` field (stored as a 4-digit military-time integer) was reformatted into `HH:MM` string representation.
5. **Categorical decoding** — Single-letter codes for `Vict Sex` (`M`, `F`, `X`, `H`) and `Vict Descent` (`A`, `B`, `C`, `H`, `W`, etc.) were mapped to their full descriptive labels using the LAPD data dictionary.
6. **Age correction** — `Vict Age = 0` was treated as a missing-value sentinel rather than a genuine age (consistent with NIBRS/LIBRS reporting conventions, where `00` denotes an unknown victim age) and excluded from age-based analyses, together with any residual non-positive values.

This pipeline is shared between the exploratory analysis script and the modeling script to ensure consistency between the two stages of the project.

## Exploratory Data Analysis

The exploratory stage (`los_angeles_data_analysis.py`) produces a set of descriptive visualizations — victim sex distribution, victim age distribution, victim age by sex, victim descent (top 10), weapons used (top 20), and case status description — saved to the `images/` directory when the script is run. The key findings from this stage are summarized below.

**Victim sex distribution.** Male victims outnumber female victims in the cleaned dataset, with a comparatively small residual category for non-binary/unspecified and intersex classifications. This asymmetry is consistent with the offense mix in the dataset, which includes a substantial share of assault- and robbery-type crimes historically associated with a higher reported male victimization rate.

**Victim age distribution.** The age distribution of victims is unimodal and right-skewed, peaking in the late twenties to mid-thirties age range and declining steadily thereafter. Very few victims are recorded at the extremes of the age range, and the smoothness of the distribution after the age-0 correction confirms that the earlier spike at `Vict Age = 0` was an artifact of missing-value encoding rather than a genuine demographic signal.

**Victim age by sex.** Overlaying the age distributions of male and female victims shows that both groups follow a broadly similar shape, but male victim counts exceed female victim counts across nearly the entire age range, with the gap most pronounced between approximately 18 and 40 years of age.

**Victim descent (top 10).** Among the top 10 reported victim descent categories, Hispanic/Latin/Mexican and White victims constitute the two largest groups, followed by Black and Other Asian victims. The remaining categories (e.g., Chinese, Japanese, Vietnamese, American Indian/Alaskan Native) contribute markedly smaller shares, reflecting both the demographic composition of Los Angeles and differences in category granularity in the source classification scheme.

**Weapons used (top 20).** Among incidents with a recorded weapon, "Strong-Arm" (bodily force) is the most frequently reported category, followed by handguns and knife-type weapons ("Knife with Blade 6 Inches or Less" and related variants). This ranking should be interpreted against the ~67% of records with no weapon recorded, which corresponds to offenses where a weapon is not applicable rather than to unarmed incidents specifically.

**Case status description.** The large majority of cases are recorded as "Invest Cont" (investigation continuing), indicating that most incidents in the dataset remain open at the time of reporting. "Adult Other" and "Adult Arrest" form the next largest categories, while juvenile-related statuses ("Juv Arrest", "Juv Other") and "Unknown" ("CC") together account for a small fraction of records — the "CC" (Unknown) category comprises only 4 records in the modeling subset and was excluded prior to classification.

## Predictive Modeling

`ml_los_angeles.py` implements a supervised classification task using a **Decision Tree Classifier** (`scikit-learn`) to predict case **`Status`** (investigation outcome) from five structured features: `AREA`, `Part 1-2` (crime severity classification), `Vict Age`, `Vict Sex`, and `Vict Descent`.

**Modeling scope.** To keep the target space tractable and the classes reasonably populated, the analysis restricts victim sex to Male/Female and victim descent to the three most frequent categories (Hispanic/Latin/Mexican, White, Black), and removes the near-empty "CC" status category. Categorical fields were integer-encoded rather than one-hot encoded, which is consistent with the use of a tree-based classifier.

**Train/test split.** An 80/20 train-test split was used with `random_state=42` for reproducibility.

### Feature Selection Comparison

Two feature sets were evaluated:

| Feature Set | Accuracy |
|---|---|
| `AREA`, `Part 1-2`, `Vict Age`, `Vict Sex`, `Vict Descent`, `Weapon Used Cd` | 0.7352 |
| `AREA`, `Part 1-2`, `Vict Age`, `Vict Sex`, `Vict Descent` | **0.7531** |

Removing `Weapon Used Cd` slightly improved accuracy, most likely because the feature's high missingness (~67% in the raw data) reduced the effective sample size and injected noise rather than predictive signal once the affected rows were excluded.

The trained decision tree structure and the resulting confusion matrix are saved to `images/decision_tree_plot.png` and `images/confusion_matrix_plot.png`, respectively, when the script is run. The confusion matrix indicates that the classifier performs well in identifying the majority class (`IC` — Invest Cont), which dominates the label distribution, but shows reduced discriminative power for minority classes such as `JA` and `JO` (juvenile-related statuses), consistent with their low representation in the training data.

## Results and Discussion

- The final model achieves **74.5–75.3% accuracy** depending on the feature set used, using only five low-dimensional categorical/numerical features.
- Given the strong class imbalance in `Status` (the majority class, "Invest Cont," represents the large majority of the filtered dataset), accuracy alone should be interpreted cautiously; it likely reflects the model's ability to predict the majority class more than genuine discrimination among all outcome categories.
- The exploratory analysis suggests that demographic and situational patterns (age, sex, weapon type, area) are non-uniform across the dataset, but a decision tree using only demographic and geographic features has limited capacity to predict investigative outcome, which is more directly driven by case-specific investigative factors not captured in this feature set (e.g., availability of evidence, witness cooperation).
- These results should be read as an exploratory baseline rather than a production-ready predictive tool.

## Limitations

- **Class imbalance** in the `Status` target was not explicitly addressed (e.g., via resampling or class weighting), which likely inflates accuracy relative to balanced-class performance.
- **Listwise deletion** during cleaning removes a non-trivial share of records with missing demographic fields, which may introduce selection bias if missingness is not random with respect to case outcome.
- **No hyperparameter tuning** was performed on the Decision Tree Classifier; default `scikit-learn` parameters were used, which may lead to overfitting (visible in the tree depth) and does not represent the model's best achievable performance.
- **Correlational, not causal.** All relationships described (e.g., between victim descent, area, and case status) are associative and should not be interpreted as evidence of causal mechanisms, including any implication about differential treatment or investigative outcomes.
- **Date coverage** in the underlying file extends only through December 2023, despite the "2020 to 2024" filename.

## Project Structure

```
.
├── images/
│   ├── sex_ratio_analysis.png
│   ├── analysis_of_ages_of_victims.png
│   ├── analysis_of_age_comparison_of_genders.png
│   ├── analysis_of_descent_of_victims_top_10.png
│   ├── analysis_of_weapon_used_top_20.png
│   ├── analysis_of_status_description.png
│   ├── decision_tree_plot.png
│   └── confusion_matrix_plot.png
├── los_angeles_data_analysis.py
├── ml_los_angeles.py
└── README.md
```

## Requirements and Usage

**Dependencies:**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Run the exploratory analysis:**

```bash
python los_angeles_data_analysis.py
```

**Run the classification pipeline:**

```bash
python ml_los_angeles.py
```

Both scripts expect the dataset at `dataset/Crime_Data_from_2020_to_2024.csv` and write generated figures to `images/`.

## License

The dataset used in this project is publicly available via [data.gov](https://catalog.data.gov/dataset/crime-data-from-2020-to-present?from_hint=eyJzb3J0IjoicG9wdWxhcml0eSJ9) / the City of Los Angeles Open Data portal; refer to that source for the dataset's original licensing terms. The code in this repository is released under the MIT License — see `LICENSE` for details.
