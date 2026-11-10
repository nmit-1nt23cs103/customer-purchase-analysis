-- Extraction queries used by the Python steps. Each block starts with `-- name:`.

-- name: clustering_input
-- desc: Customers with valid age/income and the behavioural + demographic features used by K-Means.
SELECT customer_id,
       income,
       age,
       children,
       total_spend,
       total_purchases,
       web_visits_month,
       deal_purchases,
       catalog_purchases
FROM customers
WHERE income > 0
  AND age BETWEEN 18 AND 100;

-- name: classification_dataset
-- desc: Customer table joined to the segment assigned by the clustering step.
SELECT c.*,
       s.segment_id
FROM customers AS c
JOIN segments  AS s ON s.customer_id = c.customer_id;
