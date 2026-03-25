# Data-Science-competition-MCFITB-2026
This repository contains the results of the work from the Data Science competition MCFITB 2026. This project is a time series forecasting that includes EDA, preprocessing, feature engineering and modelling.

## Project Background
This project involved data-driven analysis and predictive modeling for an insurance company in Indonesia. Kaggle-based scoring and evaluation of the trained model using metric MAPE(Mean Absolute Percentage Error) were used.

## Problem Understanding
One issue that has emerged in recent years is the increase in individual health insurance claims, which increased by 25.5% between January and June 2025 compared to the same period in 2024. This increase in claims will impact premium adjustments, potentially making individual health insurance premiums less affordable.
Therefore, data-driven analysis is needed to predict the factors that most influence claim values. This allows initiatives such as risk selection, prevention, and early detection to minimize the impact of increased claims and maintain affordable premiums.

## Objective
- Identify the factors that influence the frequency of claims, the severity of claims and the total nominal amount that must be paid by the company.
- Training a machine learning-based model to predict trends in frequency, severity, and total nominal for the period August - December 2025

## Tools
- Python notebook (Google colab)
- Exploritary Data analysis libary (Pandas,  Numpy, Matplotlib, seaborn, statsmodels, scipy)
- Preprocessing library(StandardScaler - scikit learn)
- Forecasting library (sm for SARIMAX model - statsmodels, RandomForestRegressor - scikit learn, lGBM - lightbm)
- Evaluating library (MAPE - scikitlearn)

## Dataset Information
The dataset consists of two files: "Data_Klaim.csv" and "Data_Polis.csv." This data will be processed for analysis and used to train predictive models. The following is information from these two datasets:
- A. Data_Klaim.csv: This data records insurance users who submit claims based on a unique code called a claim ID, which represents each user's policy number. This data focuses more on the claims process, recording data such as the claim amount, claim date, etc.
- B. Data_Polis.csv: This data records personal information about insurance users, such as the policy number, plan code (type of insurance used), date of birth, domicile, and policy registration date.

## Dataset Prepareration
To process the data for analysis and training the predictive model, the two datasets will be combined into a single dataset using the policy number as the foreign key in data_claims.csv and the primary key in the policy data. New features will then be created, such as the claimant's age, treatment duration, and disease category. After this process, the tabular data will be aggregated into weekly timeframes based on the claim date, creating time series data. In line with the project objective, this predicts frequency trends calculated from the number of claim IDs occurring at a specific time. In this case, weekly. The total amount spent by the company is calculated from the number of claims issued by the company within a week. Several other variables will also be aggregated weekly. Below are some of the variables used in the aggregation of tabular data into weekly time series data.

After aggregation, we obtain several variable categories for analysis.
- A. Time variable: time
  - Functional for time series analysis
  - Seasonal decomposition
  - Trend analysis
- B. Target variable: frequency, total_nominal_claim, severity 
  - Target prediction 
  - Trend analysis
- C. Demographic variable: avg_age, pct_male -
  - Analysis of the influence of demographics on the target variable
- D. Health service: mean_los (length of stay), pct_inpatient, pct_cashless
  - Analysis of the influence of health services on the target variable
- E. Hospital cost: mean_biaya_rs, median_biaya_rs
  - Analysis of the influence of hospital costs on the target variable
- D. Disease composition: pct_kanker, pct_genitourinary, pct_mata_telinga
  - Disease driver analysis 
  - Identification of diseases contributing to claim costs
- E. Hospital Location: pct_ind, pct_sg, pct_malay 
  - Comparison of claim costs across countries 
  - Identify the influence on the target variable

## Exploritary Data Analysis 
- A. Plotting target variable
  - Frequency Plot
  ![Frequency Plot 2024 - 2025](image/freq_img.png)
  - Severity Plot
  ![Severity Plot 2024 - 2025](image/sev_img.png)
- B. Moving Avarage(SMA & EMA)
  - Frequency Moving Avarage
  ![Frequency Moving Avarage Plot 2024 - 2025](image/freq_ma.png)
  - Severity Moving Avarage
  ![Severity Plot 2024 - 2025](image/sev_ma.png)
- C. Time Series Analysis
In this section, we analyze the trend and seasonality of the target variable data. The frequency and severity patterns appear quite volatile for the 2024-2025 period. The methods used for trend analysis are Simple Moving Average and Exponential Moving Average, with SMA(4) and EMA(4). This means we use a moving average with a period of 4 weeks, or approximately one month. This method helps smooth out the volatility of the data.
  - Trend Analysis:
    1. Frequency Analysis:
    The frequency values ​​are mostly in the range of 50-70 claims per week. Some spikes in the original data do not significantly change the direction of the SMA and EMA. The frequency trend tends to be stable from 2024 to mid-2025.
    2. Severity Analysis:
    The high spike in early 2025 was likely caused by a large number of claims, but the SMA and EMA lines returned to normal ranges after the spike. The severity trend was relatively stable, but there were several outliers with very high claim values.
  - Seasonal Decomposition Analysis:
    1. Frequency seasonal decomposition
    ![Seasonal Decomposition Frequency](image/seasonaldecompose_freq.png)
    2. Severity seasonal decomposition
    ![Seasonal Decomposition Severity](image/seasonaldecompose_sev.png)
