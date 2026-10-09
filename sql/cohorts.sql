SELECT date_trunc('month', signup_date) AS signup_cohort, plan, COUNT(*) AS customers,
       AVG(tenure_days) AS avg_tenure_days FROM customers GROUP BY 1,2 ORDER BY 1,2;
