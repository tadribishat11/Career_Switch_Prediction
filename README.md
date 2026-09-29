# Career Switch Prediction

## Introduction

This project predicts whether a data science job seeker will switch careers. The dataset contains candidate information such as education, work experience, company details, and training hours. The target column is `will_change_career`, with binary values 0 (no switch) and 1 (will switch).

The goal is to train and compare multiple ML models on this binary classification task. The task is useful for HR teams and training platforms that want to understand candidate intentions and make better hiring or training decisions.

**Models applied:** KNN, Decision Tree, and Neural Network (supervised). K-Means Clustering was also applied as an unsupervised analysis.



## Dataset

| Property | Details |
| --- | --- |
| Number of features | 14 total (13 input features + 1 output feature) |
| Number of data points | 5,000 rows |
| Problem type | Classification |
| Target column | `will_change_career` (0 = No Switch, 1 = Will Switch) |
| Quantitative features | `city_development_index`, `training_hours`, `will_change_career` (`enrollee_id` dropped) |
| Categorical features | `city`, `gender`, `relevent_experience`, `enrolled_university`, `education_level`, `major_discipline`, `experience`, `company_size`, `company_type`, `last_new_job` |

The target has only two discrete values, so this is a classification problem. The classes are imbalanced (74.76% Class 0 vs 25.24% Class 1).



### Feature Types

| Feature | Type | Notes |
| --- | --- | --- |
| `enrollee_id` | Quantitative | Dropped: just an identifier, not predictive |
| `city` | Categorical | 113 unique city codes |
| `city_development_index` | Quantitative | Continuous; range 0.448 to 0.949 |
| `gender` | Categorical | Male / Female / Other |
| `relevent_experience` | Categorical | Has / No relevant experience |
| `enrolled_university` | Categorical | Full time / Part time / No enrollment |
| `education_level` | Categorical | Graduate / Masters / PhD / High School / Primary School |
| `major_discipline` | Categorical | STEM / Business / Arts / Humanities / Other / No Major |
| `experience` | Categorical | Years of work experience as a text range (e.g. `>20`) |
| `company_size` | Categorical | <10 / 10-49 / 50-99 / 100-500 / 500-999 / 1000-4999 / 5000-9999 / 10000+ |
| `company_type` | Categorical | Pvt Ltd / Funded Startup / NGO / Public Sector / Early Stage / Other |
| `last_new_job` | Categorical | Years since previous job change |
| `training_hours` | Quantitative | Total training hours; range 1 to 336 |
| `will_change_career` | Quantitative | 0 = Will not switch, 1 = Will switch |



## Pre-processing

**1. Missing values**

Several columns have missing values (`company_type` 32.4%, `company_size` 31.4%, `gender` 22.3%, `major_discipline` 14.5%, and small amounts in `enrolled_university`, `education_level`, `experience`, and `last_new_job`).

- `enrollee_id` was dropped since it is only an identifier.
- Missing categorical values were filled with the mode of each column.
- Row deletion was avoided because it would remove too much data from a 5,000-row dataset.

**2. Categorical encoding**

- sklearn `LabelEncoder` was applied to all 10 categorical columns.
- One-Hot Encoding was avoided because `city` has 113 unique values, which would create 113 extra columns and make the data too sparse.

**3. Feature scaling**

- `StandardScaler` was applied (mean = 0, std = 1) for the distance-based and gradient-based models.
- The scaler was fit only on the training set and then applied to both train and test sets to prevent data leakage.



## Dataset Splitting

A stratified train-test split was used, with 80% training and 20% testing (`random_state=42`). Stratification keeps the original class ratio in both sets, which matters because the dataset is imbalanced.

| Split | Total Samples | Class 0 Count | Class 1 Count |
| --- | --- | --- | --- |
| Training set (80%) | 4,000 | 2,990 | 1,010 |
| Test set (20%) | 1,000 | 748 | 252 |



## Models

| Model | Configuration |
| --- | --- |
| K-Nearest Neighbors | k = 5, Euclidean distance (requires scaling) |
| Decision Tree | Gini impurity, `max_depth=10` (scaling not required) |
| Neural Network (MLP) | `MLPClassifier`, 3 hidden layers (128, 64, 32), ReLU, Adam, early stopping (`validation_fraction=0.1`) |
| K-Means (unsupervised) | k = 2 (chosen with the Elbow Method), visualized with 2-component PCA |

Models were evaluated using accuracy, precision, recall, AUC, confusion matrices, and ROC curves. Because the classes are imbalanced, AUC and recall are the more informative metrics.



## Conclusion

This project compared KNN, Decision Tree, and a Neural Network (MLP) for predicting whether a data science job seeker will switch careers, with K-Means clustering as a supporting unsupervised analysis. Full results are in the [project report](17_24101184_23201117.pdf).

### Key takeaways

- **Accuracy alone is misleading:** Because most candidates do not switch careers, a model can look accurate while missing the switchers it is meant to find. Recall and AUC gave a more honest picture.
- **Decision Tree overfitted:** It fit the training data much better than the test data, showing that limiting tree depth was not enough to make it generalize well.
- **Neural Network failed on the minority class:** It fell back on predicting "No Change" for nearly everyone, ignoring career switchers entirely.
- **Clusters overlap heavily:** K-Means and PCA showed some underlying structure, but the two groups overlap a lot, so candidates are hard to separate.

### Why the models struggled

- **Class imbalance:** The majority class dominated training and biased the models toward it.
- **Weak signals:** No single feature strongly predicts a career switch, so models must combine many weak signals.
- **Data quality:** Heavy missing values in company information and the high number of unique cities made the data noisy.

### Challenges faced

- **Missing data:** Rows could not be deleted without losing too much data, so mode imputation was used and may have introduced some error.
- **Categorical complexity:** Encoding 113 cities without inflating the dataset was a major trade-off, which is why Label Encoding was chosen over One-Hot Encoding.
- **Evaluation:** Picking metrics that reflect real performance on an imbalanced dataset was essential.

### Future improvements

- Handle class imbalance with oversampling (e.g. SMOTE), undersampling, or threshold tuning.
- Try ensemble methods such as Random Forest or Gradient Boosting.
- Use smarter imputation and encoding for high-cardinality features like `city`.



## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn
