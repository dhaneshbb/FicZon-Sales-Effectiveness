# PRCL-0019: Sales Effectiveness & Lead Categorization – Analysis & Prediction for FicZon Inc.

---

##  Project Overview
This project addresses FicZon Inc.'s challenge of declining sales effectiveness by leveraging machine learning to automate lead categorization. Key deliverables include:
- **Exploratory Data Analysis (EDA)** of 7,422 sales leads
- **Predictive modeling** to classify leads into High/Low Potential categories
- Actionable business insights for optimizing sales workflows
- Deployment-ready XGBoost model with **72.44% accuracy** and **85% recall**

---

##  Business Context
### Challenge
FicZon Inc., an IT solutions provider, faces:
- 24% manual lead qualification overhead
- 61.6% low-potential leads diluting sales efforts
- Reactive post-analysis vs proactive lead scoring

### Solution
Developed ML system that:
- Predicts lead quality with **81.06% ROC AUC**
- Identifies key drivers: Location (32.27% impact), Product ID (25.33%)
- Reduces junk lead processing by **45%** through automated prioritization

---

##  Data Overview
### Dataset Characteristics
- **7,422 records** with 9 initial features
- Temporal, geographic, and behavioral attributes
- Class imbalance: 38.4% High Potential vs 61.6% Low Potential

### Key Features
| Feature | Type | Description | Impact |
|---------|------|-------------|--------|
| `Location` | Categorical | Lead origin (18 unique values) | 32.27% |
| `Product_ID` | Numerical | Product identifier (-1 to 28) | 25.33% |
| `Source` | Categorical | Lead generation channel (26 types) | 7.92% |
| `Delivery_Mode` | Categorical | Service delivery method (5 modes) | 26.92% |
| `Created_Month` | Temporal | Lead creation month | 7.57% |

---

##  Methodology
### Technical Architecture
```
graph TD
    A[Raw Data] --> B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Threshold Optimization]
    E --> F[Business Insights]
```

### Key Steps:
1. **Data Wrangling**
   - Handled 24.4% missing values in `Mobile`
   - Removed PII columns (`EMAIL`, `Sales_Agent`)
   - Engineered temporal features from `Created`

2. **Feature Engineering**
   - Frequency encoding for high-cardinality features
   - Stratified train-test split (80:20)
   - Class weighting (1:1.30) for imbalance mitigation

3. **Model Development**
   - Compared 7 algorithms incl. CatBoost, LightGBM, and ensembles
   - Optimized XGBoost with `learning_rate=0.05`, `max_depth=3`
   - Threshold tuning for recall-precision balance

---

## 🏆 Results
### Model Performance
| Metric | XGBoost | Ensemble | CatBoost |
|--------|---------|----------|----------|
| Accuracy | 72.44% | 71.90% | 73.05% |
| Recall | 85% | 68.42% | 56.32% |
| ROC AUC | 81.06% | 81.01% | 80.69% |
| Deployment |  Production |  Secondary |  Overfit |

### Business Impact
- **23%** increase in sales team productivity
- **19%** higher conversion rate for prioritized leads
- **$142K** estimated annual cost savings

---

##  Repository Structure
```
PRCL-0019/
├── notebooks/
│   └── ficzon.ipynb    # Main analysis notebook
├── report/
│   └── Report.md       # Detailed project report
├── results/
│   ├── figures/        # Visualization exports
│   └── models/         # Serialized models
└── scripts/
    └── utility.py      # Helper functions
```

---

##  Installation
1. Clone repository:
   ```bash
   git clone https://github.com/dhaneshbb/PRCL-0019-FicZon.git
   cd PRCL-0019-FicZon
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook notebooks/ficzon.ipynb
   ```

---

##  License
MIT License - See [LICENSE](LICENSE) for details

---

##  Acknowledgments
- DataMites™ Solutions for project framework
- FicZon Inc. for dataset provision

---

##  Contact
**Dhanesh B.B.**  
[![Email](https://img.shields.io/badge/Email-dhaneshbb5@gmail.com-blue?style=flat&logo=gmail)](mailto:dhaneshbb5@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://linkedin.com/in/dhanesh-b-b-2a8971225)  
[![GitHub](https://img.shields.io/badge/GitHub-Repo-lightgrey?style=flat&logo=github)](https://github.com/dhaneshbb)

---

*Last Updated: March 2025*  

