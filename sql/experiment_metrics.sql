SELECT experiment_id,variant,COUNT(*) n,SUM(converted) conversions,AVG(converted::float) conversion_rate
FROM experiments GROUP BY 1,2 ORDER BY 1,2;
