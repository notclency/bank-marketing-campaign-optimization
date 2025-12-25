Here is a comprehensive, professional `README.md` for your **Bank Marketing Optimization** repository.

It highlights the **ROI-focused approach** (not just accuracy metrics) which will stand out to hiring managers like Scott (Product Lead) and Heather (Lead Data Scientist).

***

# 💰 Bank Marketing Campaign Optimization

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gradient Boosting](https://img.shields.io/badge/Model-Gradient%20Boosting-emerald?style=for-the-badge)
![Business Value](https://img.shields.io/badge/Focus-ROI%20Optimization-blue?style=for-the-badge)

A machine learning project that goes beyond simple prediction to optimize the **Net Profit** of telemarketing campaigns. By training a **Gradient Boosting Classifier** and building a custom ROI Simulator, this solution identifies high-value leads and minimizes wasted operational costs.

**[🌐 View Live ROI Simulator & Dashboard](https://clencytabe.com/projects/bank-marketing)**

---

## 📉 The Business Problem
Telemarketing is expensive. Contacting every customer in a database results in high operational costs (OpEx) and low conversion rates.
*   **The Goal:** Predict which clients will subscribe to a term deposit.
*   **The Constraint:** Maximize **Profit**, not just Accuracy. A model that is "safe" but catches few leads results in missed revenue (Opportunity Cost). A model that is "aggressive" wastes money calling uninterested people (Wasted OpEx).

---

## 🚀 The Solution

### 1. Model Selection & Benchmarking
We tested four distinct classifiers to find the best balance for tabular marketing data. **Gradient Boosting** outperformed Neural Networks due to its ability to handle non-linear relationships in categorical data (Job, Education) effectively.

| Model | Accuracy | Insight |
| :--- | :--- | :--- |
| **Gradient Boosting** | **84.6%** | **Winner.** Best generalization and handling of categorical features. |
| Voting Classifier | 80.8% | Good stability, but slightly lower peak performance. |
| Neural Network (MLP) | 78.2% | Struggled with sparse categorical data compared to tree-based models. |
| Naive Bayes | 75.1% | Fast baseline, but too simple for complex behavioral patterns. |

### 2. The ROI Simulator (Precision-Recall Tradeoff)
Instead of a static threshold, I built a logic layer to simulate financial outcomes:
*   **Precision:** If we target high precision, we reduce call costs but miss potential sales.
*   **Recall:** If we target high recall, we capture all revenue but burn budget on bad leads.
*   **Optimization:** The model allows stakeholders to adjust the decision threshold based on their specific **Cost Per Call** vs. **Revenue Per Sale**.

---

## 📊 Key Insights & Strategy

Based on Feature Importance and EDA, we derived the following actionable strategies for the marketing team:

### 1. 📞 The "Duration" Factor
Call duration is the #1 predictor (59% importance).
*   **Insight:** If a call exceeds **375 seconds**, conversion probability doubles.
*   **Action:** Script sales calls to engage clients past the 6-minute mark.

### 2. 📅 Seasonality
*   **Insight:** Activity peaks in May, but conversion rates are lowest then.
*   **Action:** Shift budget to **March, September, October, and December**, where conversion rates are highest despite lower volume.

### 3. 👥 Demographics
*   **Target:** Students (60% conversion) and Retirees (76% conversion).
*   **Avoid:** Blue-collar and Services roles (lowest conversion rates).

---

## 🛠️ Tech Stack

*   **Language:** Python 3.9+
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
*   **Algorithms:** Gradient Boosting, Random Forest, MLPClassifier
*   **Frontend Visualization:** React, Tailwind CSS, Framer Motion (for the Portfolio Dashboard)

---

## 📂 Dataset
The dataset is the **Bank Marketing Data Set** from the UCI Machine Learning Repository.
*   **Input:** 45,000+ records (Age, Job, Marital Status, Education, Balance, Housing Loan, etc.)
*   **Target:** `y` (Has the client subscribed a term deposit?)

---

## 💻 Running Locally

### Prerequisites
*   Python 3.8+
*   Jupyter Notebook

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/notclency/bank-marketing-campaign-optimization.git
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Run the notebook:
    ```bash
    jupyter notebook notebooks/Bank_Marketing_Analysis.ipynb
    ```

---

## 📬 Contact
**Clency Tabe**
Data Science & Computer Science Student
[LinkedIn](https://linkedin.com/in/clency-tabe) | [Portfolio](https://clencytabe.com)
