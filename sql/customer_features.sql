WITH event_features AS (
 SELECT customer_id, COUNT(*) FILTER (WHERE event_ts >= TIMESTAMP '2026-09-01') AS frequency_30d,
        EXTRACT(DAY FROM TIMESTAMP '2026-09-30' - MAX(event_ts)) AS recency_days,
        COUNT(*) FILTER (WHERE event_type='feature_use')::float / NULLIF(COUNT(*),0) AS engagement_score
 FROM events GROUP BY customer_id
), tx_features AS (
 SELECT customer_id, COALESCE(SUM(amount) FILTER (WHERE transaction_ts >= TIMESTAMP '2026-09-01'),0) AS monetary_30d
 FROM transactions GROUP BY customer_id
)
SELECT c.customer_id,c.plan,c.region,c.tenure_days,
       COALESCE(e.recency_days,30) recency_days,COALESCE(e.frequency_30d,0) frequency_30d,
       COALESCE(t.monetary_30d,0) monetary_30d,COALESCE(e.engagement_score,0) engagement_score
FROM customers c LEFT JOIN event_features e USING(customer_id) LEFT JOIN tx_features t USING(customer_id);
