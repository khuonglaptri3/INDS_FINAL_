### Folder Structure 
```bash
data/
├── processed/
│   ├── adult_after_eda.csv
│   ├── adult_features_test.csv
│   ├── adult_features_train.csv
│   └── mapping_.csv
├── raw/
│   ├── adult.csv
│   └── OGHHST.xls
└── Readme.md
```
##  Raw Data Description

**`raw/`**: This directory contains the original datasets used in the project.

- Includes: `adult.csv`, `OGHHST.xls` .

- **`adult.csv`**:  
  This is the **Census Income dataset** sourced from the U.S. Census Bureau.  
  Each observation represents an individual, where each row contains demographic and financial information about a person.

- The dataset consists of **48,842 independent observations** with **15 variables**, including:
  - **14 features**
  - **1 target variable** (`income`)
- Linkdataset : https://archive.ics.uci.edu/dataset/2/adult

#  Adult Dataset Description

| Variable Name     | Role    | Type        | Demographic     | Description                                                                                                                                                          | Units | Missing Values |
|------------------|---------|-------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|----------------|
| age              | Feature | Integer     | Age             | Age of individual                                                                                                                                                    |       | No             |
| workclass        | Feature | Categorical | Income          | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked                                                              |       | Yes            |
| fnlwgt           | Feature | Integer     |                 | Final weight                                                                                                                                                         |       | No             |
| education        | Feature | Categorical | Education Level | Bachelors, HS-grad, Masters, Doctorate, etc.                                                                                                                         |       | No             |
| education-num    | Feature | Integer     | Education Level | Numerical representation of education level                                                                                                                          |       | No             |
| marital-status   | Feature | Categorical | Other           | Married, Divorced, Never-married, Separated, Widowed, etc.                                                                                                          |       | No             |
| occupation       | Feature | Categorical | Other           | Tech-support, Sales, Exec-managerial, Machine-op, etc.                                                                                                               |       | Yes            |
| relationship     | Feature | Categorical | Other           | Wife, Husband, Own-child, Not-in-family, etc.                                                                                                                        |       | No             |
| race             | Feature | Categorical | Race            | White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other                                                                                                          |       | No             |
| sex              | Feature | Binary      | Sex             | Female, Male                                                                                                                                                         |       | No             |
| capital-gain     | Feature | Integer     |                 | Capital gains                                                                                                                                                        |       | No             |
| capital-loss     | Feature | Integer     |                 | Capital losses                                                                                                                                                       |       | No             |
| hours-per-week   | Feature | Integer     |                 | Number of working hours per week                                                                                                                                     |       | No             |
| native-country   | Feature | Categorical | Other           | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, India, Japan, Greece, China, Vietnam, Mexico, etc.                                                  |       | Yes            |
| income           | Target  | Binary      | Income          | >50K, <=50K                                                                                                                                                         |       | No             |
 
##  OGHHST.xls Description

- **OGHHST.xls** contains World Bank data on **GNI (Gross National Income) per capita thresholds** and **country income classifications**.

- The dataset provides **annual thresholds** used to classify countries into income groups:
  - Low income  
  - Lower middle income  
  - Upper middle income  
  - High income  

- It also includes **operational lending categories** used by the World Bank (e.g., IDA, IBRD classifications), which determine financing eligibility.

- The data spans multiple fiscal years and reflects changes in thresholds over time.

- This file can be used to:
  - Map countries into income groups  
  - Analyze economic development levels  
  - Support feature engineering (e.g., grouping countries by income level)
- link: https://databankfiles.worldbank.org/public/ddpext_download/site-content/OGHIST.xls

##  Processed Data Description

**`Processed/`**: This directory contains the processed datasets used in the project.
Includes: `adult_after_eda.csv`, `adult_features_test.csv`, `adult_features_train.csv`, `mapping.csv`.
##  adult_after_eda Description

- This dataset is derived from `adult.csv` after initial data preprocessing and exploratory data analysis (EDA).

- Missing values have been handled and cleaned.

- Two columns have been removed:
  - `education-num`
  - `fnlwgt`

- The dataset serves as the **intermediate cleaned version** before feature engineering.
##  adult_features_train.csv & adult_features_test.csv Description

These datasets are the **final outputs of the feature engineering pipeline**.

- **`adult_features_train.csv`**:  
  Used for training machine learning models. 80% splited from orginal dataset.

- **`adult_features_test.csv`**:  
  Used for evaluating model performance on unseen data. 20% from orginal dataset.

###  Key Characteristics

- Contain **fully processed features**, including:
  - Nonlinear transformations (Yeo-Johnson, Robust Scaling)
  - Encoded categorical variables (Ordinal, CatBoost, LOO, One-Hot, Binary)
  - Econometric interaction features
  - Fairness-aware interaction features

- All variables are **numerical and model-ready** (no raw categorical values remain).
- - The dataset is split into **training and testing sets**, with **80% of the data used for training** and **20% for testing**, based on the original dataset.

- The dataset preserves the original prediction task:
  - **Target variable**: `income` (Binary classification)

- The split between train and test ensures:
  - No data leakage
  - Reliable evaluation of model generalization

###  Purpose

These datasets serve as the **input for model training, validation, and fairness analysis**, enabling:
- Predictive modeling (classification of income level)
- Feature importance analysis
- Bias and fairness evaluation across demographic groups
##  Feature Description

| Variable | Type | Description |
|----------|------|-------------|
| age | Numeric | Age of the individual |
| workclass | Encoded | Employment type (encoded from original categories) |
| education-num | Numeric | Years of education (proxy for education level) |
| capital-gain | Numeric | Capital gains (Yeo-Johnson transformed) |
| capital-loss | Numeric | Capital losses (Yeo-Johnson transformed) |
| hours-per-week | Numeric | Weekly working hours (robust scaled) |
| income | Binary (Target) | Income level: 1 (>50K), 0 (≤50K) |

---

###  Country & Occupation Features

| Variable | Type | Description |
|----------|------|-------------|
| country_income_group | Ordinal | Country grouped by World Bank income level |
| occupation_group | Encoded | Occupation grouped into skill-based categories |
| married_flag | Binary | Marital status: 1 (married), 0 (not married) |

---

###  Relationship (One-Hot)

| Variable | Description |
|----------|-------------|
| relationship_Husband | Individual is a husband |
| relationship_Not-in-family | Not part of a nuclear family |
| relationship_Other-relative | Other family relation |
| relationship_Own-child | Individual is a child |
| relationship_Unmarried | Unmarried individual |
| relationship_Wife | Individual is a wife |

---

###  Race (One-Hot)

| Variable | Description |
|----------|-------------|
| race_Amer-Indian-Eskimo | American Indian / Eskimo |
| race_Asian-Pac-Islander | Asian / Pacific Islander |
| race_Black | Black |
| race_Other | Other race |
| race_White | White |

---

###  Gender

| Variable | Type | Description |
|----------|------|-------------|
| sex_binary | Binary | Gender: 1 (Male), 0 (Female) |

---

###  Econometric Interaction Features

| Variable | Formula | Description |
|----------|---------|-------------|
| human_capital | age × education-num | Accumulated human capital over time |
| household_labour | hours-per-week × married_flag | Labor supply intensity for married individuals |
| net_capital | capital-gain − capital-loss | Net financial capital |

---

###  Fairness Interaction Features (by Race)

#### Education × Race

| Variable | Description |
|----------|-------------|
| edu_x_Amer-Indian-Eskimo | Education return for Amer-Indian-Eskimo group |
| edu_x_Asian-Pac-Islander | Education return for Asian-Pac-Islander group |
| edu_x_Black | Education return for Black group |
| edu_x_Other | Education return for Other race group |
| edu_x_White | Education return for White group |

#### Hours × Race

| Variable | Description |
|----------|-------------|
| hours_x_Amer-Indian-Eskimo | Work hours burden for Amer-Indian-Eskimo group |
| hours_x_Asian-Pac-Islander | Work hours burden for Asian-Pac-Islander group |
| hours_x_Black | Work hours burden for Black group |
| hours_x_Other | Work hours burden for Other race group |
| hours_x_White | Work hours burden for White group |

#### Capital × Race

| Variable | Description |
|----------|-------------|
| capital_x_Amer-Indian-Eskimo | Capital access for Amer-Indian-Eskimo group |
| capital_x_Asian-Pac-Islander | Capital access for Asian-Pac-Islander group |
| capital_x_Black | Capital access for Black group |
| capital_x_Other | Capital access for Other race group |
| capital_x_White | Capital access for White group |
##  mapping_.csv Description

- This file provides a mapping of countries to **income groups (year 1994)**.

- It is used during feature engineering to transform the `native-country` variable into a more meaningful feature: `country_income_group`.

- Country names are standardized and aligned with the Adult dataset using reference data from `OGHHST.xls`.

###  Structure

| Column | Description |
|--------|-------------|
| country_adult_dataset | Country name as appears in the Adult dataset |
| income_group | Income classification based on World Bank categories |

###  Income Groups

- **L**: Low income  
- **LM**: Lower middle income  
- **UM**: Upper middle income  
- **H**: High income  

###  Purpose

- Reduce high-cardinality categorical variable (`native-country`)
- Incorporate macroeconomic context into the model
- Improve generalization and interpretability during feature engineering