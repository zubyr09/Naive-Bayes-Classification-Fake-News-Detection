# Naive Bayes Classification for Fake News Detection

**Author:** [zubyr09](https://github.com/zubyr09)
**Repository:** [Naive-Bayes-Classification-Fake-News-Detection](https://github.com/zubyr09/Naive-Bayes-Classification-Fake-News-Detection)
**Notebook:** [Naive_Bayes_Classification_Fake_News_Detection.ipynb](https://github.com/zubyr09/Naive-Bayes-Classification-Fake-News-Detection/blob/main/Naive_Bayes_Classification_Fake_News_Detection.ipynb)
**Dataset:** [Assignment_Data_fake_or_real_news.csv.gz](https://github.com/zubyr09/Naive-Bayes-Classification-Fake-News-Detection/blob/main/Assignment_Data_fake_or_real_news.csv.gz)

## Project Overview

This project focuses on developing and evaluating Naive Bayes classifiers to distinguish between "REAL" and "FAKE" news articles. Utilizing a dataset of news titles and text content, the primary objective is to apply a structured Natural Language Processing (NLP) and machine learning workflow. This includes comprehensive data preprocessing, feature extraction using Bag of Words and TF-IDF techniques, training multiple Naive Bayes variants, and conducting a thorough evaluation of their performance. The project emphasizes clarity, professionalism, and modern data science practices.

## Table of Contents

1.  [Objective](#objective)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
    * [Setup & Libraries](#setup--libraries)
    * [Data Loading & Initial Exploration](#data-loading--initial-exploration)
    * [Data Preprocessing](#data-preprocessing)
    * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    * [Train-Test Split](#train-test-split)
    * [Feature Extraction (Vectorization)](#feature-extraction-vectorization)
    * [Model Training](#model-training)
4.  [Model Evaluation & Results](#model-evaluation--results)
    * [Performance Metrics](#performance-metrics)
    * [Key Findings](#key-findings)
5.  [Prediction on New Data](#prediction-on-new-data)
6.  [How to Run](#how-to-run)
7.  [Conclusion & Further Improvements](#conclusion--further-improvements)

## 1. Objective
<a name="objective"></a>
The main goal of this project is to build an effective text classification model using Naive Bayes algorithms to identify fake news articles. This involves comparing different text vectorization techniques and Naive Bayes variants to determine the most suitable approach for this specific dataset.

## 2. Dataset
<a name="dataset"></a>
The dataset used is `Assignment_Data_fake_or_real_news.csv.gz`. It contains the following columns:
* `Unnamed: 0`: An index column (dropped during preprocessing).
* `title`: The title of the news article.
* `text`: The main content of the news article.
* `label`: The target variable, indicating whether the news is 'REAL' or 'FAKE'.

The dataset is relatively balanced between 'REAL' and 'FAKE' news articles.

## 3. Methodology
<a name="methodology"></a>

### 3.1. Setup & Libraries
<a name="setup--libraries"></a>
The project utilizes standard Python libraries for data manipulation, NLP, machine learning, and visualization:
* Core: `pandas`, `numpy`, `re`, `string`
* Visualization: `matplotlib`, `seaborn`, `wordcloud`
* NLP: `nltk` (for stopwords, tokenization, lemmatization)
* Machine Learning: `scikit-learn` (for train-test split, vectorizers, Naive Bayes models, metrics, pipelines)

Dark-themed visualizations are configured for better readability and a modern aesthetic.

### 3.2. Data Loading & Initial Exploration
<a name="data-loading--initial-exploration"></a>
The dataset is loaded from the `.csv.gz` file. Initial exploration includes:
* Displaying the first few rows (`df.head()`).
* Getting dataset information (`df.info()`).
* Checking the shape and identifying missing values (`df.isnull().sum()`).

### 3.3. Data Preprocessing
<a name="data-preprocessing"></a>
A comprehensive preprocessing pipeline is applied to the text data:
* **Column Dropping:** The initial 'Unnamed: 0' index column is removed.
* **Missing Value Handling:** NaN values in 'title' and 'text' are filled with empty strings. Rows with missing 'label' or entirely empty 'text' (after stripping whitespace) are dropped.
* **Feature Combination:** The 'title' and 'text' columns are combined into a single 'full_text' feature for a richer representation of the news content.
* **Target Variable Encoding:** The 'label' column is numerically encoded: 'FAKE' is mapped to `1` and 'REAL' to `0`.
* **Advanced Text Cleaning (`preprocess_text_advanced` function):**
    * Lowercase conversion.
    * Removal of URLs, user @mentions, and hashtags.
    * Punctuation removal.
    * Number removal.
    * Tokenization using `nltk.word_tokenize`.
    * Stopword removal using NLTK's English stopword list (with added custom stopwords like 'said', 'also', 'would').
    * Lemmatization using `nltk.WordNetLemmatizer` to reduce words to their base form.

### 3.4. Exploratory Data Analysis (EDA)
<a name="exploratory-data-analysis-eda"></a>
* **Label Distribution:** A Seaborn countplot visualizes the distribution of 'REAL' vs. 'FAKE' news labels, showing percentages for each class. The dataset is fairly balanced.
* **Word Clouds:** Word clouds are generated for both 'REAL' and 'FAKE' news based on the `processed_text`. This provides a visual indication of the most frequent and potentially differentiating terms for each class.

### 3.5. Train-Test Split
<a name="train-test-split"></a>
The preprocessed dataset (`processed_text` as features and `label_encoded` as the target) is split into training (75%) and testing (25%) sets. `stratify=y` is used to ensure similar class proportions in both splits.

### 3.6. Feature Extraction (Vectorization)
<a name="feature-extraction-vectorization"></a>
Two common text vectorization techniques are employed:
* **CountVectorizer (Bag of Words):**
    * Parameters: `ngram_range=(1, 2)` (to include unigrams and bigrams), `stop_words='english'`, `max_df=0.7` (ignore terms appearing in >70% of documents), `min_df=3` (ignore terms appearing in <3 documents).
* **TfidfVectorizer (Term Frequency-Inverse Document Frequency):**
    * Parameters: `ngram_range=(1, 2)`, `stop_words='english'`, `max_df=0.7`, `min_df=3`, `use_idf=True`, `smooth_idf=True`, `sublinear_tf=True`. TF-IDF often yields better results by weighting terms based on their importance.

### 3.7. Model Training
<a name="model-training"></a>
Naive Bayes classifiers are trained using scikit-learn `Pipeline` to combine vectorization and classification:
* **Multinomial Naive Bayes (`MultinomialNB`)** with `CountVectorizer`.
* **Multinomial Naive Bayes (`MultinomialNB`)** with `TfidfVectorizer`.
* **Complement Naive Bayes (`ComplementNB`)** with `TfidfVectorizer`. ComplementNB is often effective for text classification and can handle imbalanced datasets well (though this dataset is relatively balanced).

A smoothing parameter `alpha=0.1` (Laplace smoothing) is used for all Naive Bayes models.

## 4. Model Evaluation & Results
<a name="model-evaluation--results"></a>

### 4.1. Performance Metrics
<a name="performance-metrics"></a>
Models are evaluated on the test set using:
* **Accuracy:** Overall correctness of predictions.
* **Classification Report:** Provides precision, recall, and F1-score for each class ('REAL', 'FAKE').
* **Confusion Matrix:** Visualized using `seaborn.heatmap` to show true positives, true negatives, false positives, and false negatives.
* **ROC Curve and AUC Score:** Plotted for all models to compare their ability to distinguish between classes across different thresholds.

### 4.2. Key Findings
<a name="key-findings"></a>
The performance of the trained models on the test set was as follows:

| Model                             | Accuracy | REAL (0) Precision | REAL (0) Recall | REAL (0) F1-Score | FAKE (1) Precision | FAKE (1) Recall | FAKE (1) F1-Score | ROC AUC |
| :-------------------------------- | :------- | :----------------- | :-------------- | :---------------- | :----------------- | :-------------- | :---------------- | :------ |
| MultinomialNB + CountVectorizer   | ~0.9167  | 0.90               | 0.94            | 0.92              | 0.94               | 0.89            | 0.91              | ~0.97   |
| MultinomialNB + TfidfVectorizer   | ~0.9236  | 0.90               | 0.95            | 0.93              | 0.95               | 0.89            | 0.92              | ~0.98   |
| **ComplementNB + TfidfVectorizer**| **~0.9261**| **0.91** | **0.95** | **0.93** | **0.95** | **0.90** | **0.92** | **~0.98**|

* **ComplementNB with TfidfVectorizer** yielded the highest accuracy and generally strong precision/recall for both classes.
* TF-IDF vectorization consistently outperformed CountVectorizer.
* All models demonstrated a good ability to distinguish between REAL and FAKE news, as indicated by high ROC AUC scores (around 0.97-0.98).

The classification reports show that the models are effective at identifying both REAL and FAKE news, with F1-scores above 0.90 for both classes with the best model. Confusion matrices visually confirm these findings, showing the distribution of correct and incorrect predictions.

## 5. Prediction on New Data
<a name="prediction-on-new-data"></a>
The notebook demonstrates how the best-performing pipeline (`ComplementNB with TfidfVectorizer`) can be used to predict the label (REAL/FAKE) and probabilities for new, unseen news articles. This involves applying the same text preprocessing steps to the new data before feeding it to the model.

## 6. How to Run
<a name="how-to-run"></a>

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zubyr09/Naive-Bayes-Classification-Fake-News-Detection.git](https://github.com/zubyr09/Naive-Bayes-Classification-Fake-News-Detection.git)
    cd Naive-Bayes-Classification-Fake-News-Detection
    ```
2.  **Ensure Dependencies are Installed:**
    The notebook requires Python 3 and the libraries listed in [Section 3.1](#setup--libraries). You can typically install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn tqdm
    ```
3.  **Download NLTK Resources:**
    Run the NLTK download cells within the notebook if you haven't downloaded `stopwords`, `punkt`, and `wordnet` before.
4.  **Dataset:**
    The dataset `Assignment_Data_fake_or_real_news.csv.gz` should be in the same directory as the notebook. It will be loaded automatically.
5.  **Run Jupyter Notebook:**
    Open and run the `Naive_Bayes_Classification_Fake_News_Detection.ipynb` file using Jupyter Notebook or JupyterLab.

## 7. Conclusion & Further Improvements
<a name="conclusion--further-improvements"></a>
This project successfully demonstrated the application of Naive Bayes classifiers for fake news detection. The `ComplementNB` model combined with `TfidfVectorizer` and advanced text preprocessing (including lemmatization and n-grams) achieved a commendable accuracy of approximately 92.61%. The comprehensive evaluation, including ROC curves and confusion matrices, confirms the model's effectiveness.

**Potential Further Improvements:**
* **Hyperparameter Tuning:** Employ `GridSearchCV` or `RandomizedSearchCV` to find optimal parameters for vectorizers and Naive Bayes classifiers.
* **Advanced Text Preprocessing:** Experiment with different stemming algorithms, more sophisticated stopword lists, or techniques like Part-of-Speech (POS) tagging for more nuanced lemmatization.
* **Feature Engineering & Selection:**
    * Explore word embeddings (e.g., Word2Vec, GloVe, FastText) or transformer-based embeddings (e.g., BERT) for richer semantic representations, although these are typically used with more complex models than Naive Bayes.
    * Implement feature selection techniques if using a very large number of features.
* **Different Classifiers:** Compare Naive Bayes with other text classification algorithms like Logistic Regression, Support Vector Machines (SVM), or ensemble methods (Random Forest, Gradient Boosting).
* **Error Analysis:** Conduct a thorough analysis of misclassified instances to understand model weaknesses.
* **Handling Imbalance (if more pronounced):** If the dataset were highly imbalanced, techniques like SMOTE or using different class weights could be beneficial.
* **Cross-Validation Strategy:** Implement k-fold cross-validation during model selection and hyperparameter tuning for more robust performance estimates.
* **Interpretability:** For more complex models, utilize LIME or SHAP for explaining individual predictions.

This project serves as a strong foundation for text classification tasks and highlights the importance of a methodical approach, from data cleaning to model evaluation and interpretation.
