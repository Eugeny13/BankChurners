# BankChurners
Complete Bank Churners Analysis (EDA + Model)  This notebook loads data, does EDA, builds models (Logistic Regression, Random Forest), evaluates performance, and exports results (scores, importance, saved model).
## Summary:

### Data Analysis Key Findings

*   The top 10 features identified by the Random Forest model as most important for predicting churn are: `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Revolving_Bal`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio`, `Months_Inactive_12_mon`, `Total_Relationship_Count`, `Credit_Limit`, `Avg_Open_To_Buy`, and `Customer_Age`.
*   High churn clients (score\_churn \$\ge\$ 0.5) show significantly lower average values for `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Revolving_Bal`, `Total_Ct_Chng_Q4_Q1`, and `Avg_Utilization_Ratio` compared to the overall client base.
*   High churn clients have a slightly higher average for `Months_Inactive_12_mon` compared to all clients.
*   Visualizations confirm that the distributions of key features like `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Revolving_Bal`, `Avg_Utilization_Ratio`, `Total_Ct_Chng_Q4_Q1`, `Months_Inactive_12_mon`, and `Total_Relationship_Count` are notably different between high and low churn clients. High churn clients tend to be concentrated at lower values for transaction-related metrics, revolving balance, utilization ratio, and relationship count, while showing a slight shift towards higher values for months inactive.

### Insights or Next Steps

*   Focus retention efforts on clients exhibiting the characteristics identified for high churn risk, particularly those with low transaction activity, low revolving balances, and fewer relationships.
*   Investigate the root causes behind the observed patterns in the top churn-predictive features for high-risk clients. For example, understanding why transaction amounts and counts are low for these clients could inform targeted interventions.
