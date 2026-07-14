# ZeroStart-Fraud: Unsupervised Anomaly Detection & Survival Analytics

A cold-start fraud detection system that uses unsupervised machine learning to spot anomalies without historical labels. It links model errors directly to user retention using survival analysis.

## Problem Overview
Traditional supervised fraud detection models require weeks of historical label collection and data integration, exposing companies to massive losses during the "Cold Start" window. This system addresses that gap by establishing transactional security baselines on Day One. Crucially, it accounts for the business trade-off: false negatives result in fraud losses, while false positives cause user friction and customer churn.

## Architecture & Tech Stack
- **Ingestion & Features:** Continuous chronological ordering, rolling temporal density engineering using `Pandas` and `DuckDB`.
- **Core AI Engine:** Unsupervised `Isolation Forest` configured for high-dimensional, unlabeled anomalies.
- **Business Translation Layer:** `Cox Proportional Hazards` survival modeling to quantify user churn risk driven by model misclassifications.

## Quickstart Guide
1. Clone this repository structure into your local environment.
2. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt