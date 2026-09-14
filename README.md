# Data Encoding and Preprocessing for House Price Predictions

## Overview

This project prepares categorical and numeric data for machine learning on the
Ames housing dataset (`train.csv` / `test.csv`, target: `SalePrice`). It covers
the full pipeline of  detecting categorical columns, handling missing values,
engineering new features, applying eight different encoding techniques, and
comparing them across three regression models with cross validation.

## 1. Identifying Categorical Columns (the how)

Instead of manually listing column names, categorical columns are detected
directly from the DataFrame's dtypes using `pd.api.types.is_string_dtype`
rather than a plain `dtype == 'object'` check. This matters
because it correctly identifies columns as categorical even when they still
contain missing values. Plain `object` checks can miss these depending on
the pandas version, silently dropping columns from the pipeline.

This makes preprocessing more flexible and lets it keep working if the
dataset changes.

## 2. Handling Missing Values (when you can not find)

Missing categorical values are handled using information from
`data_description.txt`. Columns where `NaN` indicates **"feature not present"**
are stored in a dedicated list
(`NA_MEANS_NONE_COLUMNS`, 15 columns) and filled with the string `'None'` using
a single reusable function, avoiding repetitive `fillna()` calls for each
column. Any remaining missing categorical values are filled with that
column's mode.

This approach is used to simplify maintenance and keeps the preprocessing pipeline
easy to update as the dataset changes.

## 3. Feature Engineering

Beyond encoding, a set of domain features is derived from the raw columns to
help models pick up relationships that are not obvious from individual raw
columns.

- **`TotalSF`** — combined basement + 1st floor + 2nd floor square footage
- **`HouseAge`** / **`RemodAge`** — years since last remodel/year built
- **`WasRemodeled`** — flag for whether remodel year differs from build year
- **`TotalBath`** — full baths + half baths across main and
  basement levels

Two outliers documented (very large `GrLivArea` with unexpectedly low
`SalePrice`) are removed from the training set before fitting.

## 4. Encoding Techniques

Eight encoding techniques were implemented and compared

### Label Encoding
Assigns a unique number to each category (e.g. `0, 1, 2`) without creating
new columns. Numbers may imply an order that does not exist, this is mainly
suitable for tree based models.

### One Hot Encoding
Creates a new binary column for each category (e.g. `RoofStyle_Gable = 1`).
Prevents false ordinal relationships but increases dimensionality sharply
with high cardinality columns.

### Ordinal Encoding
Assigns numbers based on a meaningful order (e.g. `Po < Fa < TA < Gd < Ex`).
The same quality scale is reused across all quality rating columns
(`ExterQual`, `KitchenQual`, `BsmtQual`, etc.), with a `'None'` category
prepended for columns where absence is valid.

### Frequency / Count Encoding
Replaces each category with how often it appears in the data.
Cheap, keeps dimensionality flat, and gives the model a rough
signal of how common a category is.

### Feature Hashing
Hashes each row's categorical values into a fixed size numeric vector,
avoiding the column explosion of one hot encoding for high cardinality data,
at the cost of some information loss from hash collisions.

### Cyclic Encoding
Transforms cyclical features (`MoSold`) into sine pairs so that,
information like December and January are recognized as close together rather than far
apart on a linear scale.

### Target Encoding 
Replaces each category with the mean target value for that category (leaky).


### K-Fold Target Encoding
The leakage safe version of target encoding. for each fold, category means
are computed only from the other folds, so no row ever sees its own target
value baked into its encoding during cross validation.

## 5. Encoding Pathway Comparison

| Encoding | Best Used For |
|---|---|
| Label Encoding | Tree based models, ordinal agnostic |
| One Hot Encoding | Low cardinality categories with no natural order |
| Ordinal Encoding | Categories with an inherent order (quality ratings) |
| Frequency Encoding | High cardinality columns, cheap dimensionality |
| Feature Hashing | Very high cardinality columns, memory constrained settings |
| Cyclic Encoding | Naturally cyclical features (month, day, hours) |
| Target Encoding (leaky) | Comparison baseline only and not for production |
| K-Fold Target Encoding | High cardinality columns, production safe target signal |

All eight pathways are evaluated against three models, Linear Regression,
Random Forest, and HistGradientBoosting. 

**Result:** with engineered features included, the strongest honest
(non leaky) result was **K-Fold Target Encoding + Linear Regression**
(CV RMSE ≈ 0.118), with Label Encoding and Feature Hashing close behind.
The  leaky target encoding scored marginally better in cross validation
p because of the leakage, not because it generalizes better.

## 6. Final Predictions

The winning pathway (K-Fold Target Encoding, fits on the full training set)
is used to train the final Linear Regression model and generate predictions
for every row in `test.csv`, saved to `predicted_house_prices.csv`.

## Conclusion

The preprocessing pipeline is designed to be **flexible, reusable, and easy
to maintain**. By automatically detecting columns, handling missing values
using configuration, engineering domain informed features, and systematically
comparing eight encoding strategies against three models with proper
cross validation, the workflow adapts smoothly to dataset variations while
guarding against the kind of data leakage that can make an approach look
better than it actually is.
