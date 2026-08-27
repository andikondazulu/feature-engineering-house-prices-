# Data Encoding and Preprocessing

## Overview

This project focuses on preparing categorical data for machine learning. It includes detecting categorical columns, handling missing values, and applying different encoding techniques depending on the type of categorical variable.

## 1. Identifying Categorical Columns (The how)

Instead of manually listing column names, categorical columns are detected directly from the DataFrame's data types.

This makes the preprocessing more flexible and allows it to continue working if the dataset changes slightly.

## 2. Handling Missing Values (when you can not find)

Missing categorical values are handled using information from `data_description.txt`.

Columns where `NaN` indicates **"feature absent"** are stored in a dedicated list. A single function is used to replace these missing values, avoiding repetitive `fillna()` calls for each column.

This approach simplifies maintenance and updates to the preprocessing pipeline.

## 3. Encoding Techniques

### Label Encoding

Assigns a unique number to each category (e.g., `0, 1, 2`) without creating new columns.

**Note:** Numbers may imply an order that doesn't exist, so it's mainly suitable for tree-based models or binary features.

### One-Hot Encoding

Creates a new binary column for each category.

**Example:**

`RoofStyle = Gable` → `RoofStyle_Gable = 1`

This method prevents false ordinal relationships but can lead to high dimensionality when categories are numerous.

### Ordinal Encoding

Assigns numbers based on a meaningful order.

**Example quality scale:**

`Po < Fa < TA < Gd < Ex`

The same scale is reused across columns with similar rankings.

## 4. Why These Methods Are Used

Different types of categorical variables benefit from different encoding strategies:

| Encoding         | Best Used For                          |
| Label Encoding   | Binary categories or tree-based models |
| One-Hot Encoding | Categories with no natural order       |
| Ordinal Encoding | Categories with an inherent order      |

## Conclusion

The preprocessing pipeline is designed to be **flexible, reusable, and easy to maintain**. By automatically detecting columns, handling missing values via configuration, and choosing suitable encoding methods, the workflow adapts smoothly to dataset variations with minimal adjustments.
