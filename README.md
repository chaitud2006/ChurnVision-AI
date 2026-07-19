# ChurnVision AI — Intelligent Customer Retention Platform

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask--Runtime-emerald.svg)](https://flask.palletsprojects.com/)
[![ML-Engine](https://img.shields.io/badge/Inference--Engine-XGBoost%20%7C%20Scikit--Learn-blueviolet.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ChurnVision AI is an enterprise-grade predictive analytics platform engineered to diagnose customer attrition risks and automate retention strategies. By pairing high-performance gradient-boosted machine learning classifiers with a secure web-based diagnostic terminal, the platform translates complex multi-variant data models into actionable, high-visibility corporate mitigation workflows.

---

## ⚡ Architectural Core Components

*   **Role-Based Access Control Architecture (RBAC):** Implements an upstream web session interception layer backed by cryptographic user authentication, protecting corporate endpoints from unauthenticated exposure.
*   **Production Analytics Matrix:** Features live validation benchmarking across distinct algorithmic models (XGBoost, Random Forest, Logistic Regression) to ensure transparent precision tracking.
*   **High-Visibility Diagnostic Reporting UI:** Crafted using modern UX/UI accessibility patterns with strict color-contrast boundaries, presenting clean input matrices alongside automated prescriptive playbooks.
*   **Decoupled Functional Pipelines:** Fully modular operational scripts isolate the data ingestion, feature scaling, model optimization, and interface serving layers for maximum maintainability.

---

## 📂 Project Repository Directory Blueprint

```text
CHURNVISION-AI/
├── dataset/                   # Static data warehouse (Source data frames)
├── models/                    # Serialized production-ready model binaries (.pkl)
├── static/                    # Asset compilation node (Custom CSS, scripts, UI iconography)
├── templates/                 # Render-ready HTML5 view components
│   ├── login.html             # Secure gate for user workspace access authentication
│   ├── signup.html            # Gateway for registering encrypted access credentials
│   ├── dashboard.html         # Enterprise executive analytics hub & benchmarking matrix
│   ├── predict_form.html      # Direct predictive inference parameter submission sheet
│   └── prediction.html        # High-visibility customer risk profile diagnostic report
├── uploads/                   # Staging area for batch file input workloads
├── app.py                     # Primary micro-service entry point and controller map
├── clean_data.py              # Data cleaning, normalization, and structural auditing
├── clue_importance.py         # Computational logic for feature weights and attribution
├── compare_models.py          # Validation engine for multi-architecture benchmarking
├── eda.py                     # Visual data profiling and metric distribution manager
├── fetch_data.py              # Automated asset sourcing and data ingestion layer
├── prepare_features.py        # Numerical transformations, encoding, and vector alignment
├── train_model.py             # Optimization and compilation routine for model serialization
└── users.db                   # Isolated relational SQLite storage for secure user records
