-- monkeybeat
-- name: get-random-stocks
-- Fetch a list of random stocks for the given count.
-- $1: count
SELECT groupArraySample($1)(tradingsymbol) AS stocks FROM monkeybeat.prices WHERE segment='EQ'

-- name: get-returns
-- Get average returns for given date and given list of stocks.
-- $1: table
-- $2: date
-- $3: symbols
WITH
start AS (
	SELECT date FROM monkeybeat.prices WHERE toDate(date)>=today() - INTERVAL $2 DAY ORDER BY date ASC LIMIT 1
),
end AS (
	SELECT date FROM monkeybeat.prices WHERE toDate(date)>=today() - INTERVAL $2 DAY ORDER BY date DESC LIMIT 1
),
old AS
(
	SELECT
		close,
		tradingsymbol
	FROM monkeybeat.prices AS sp
	INNER JOIN start ON sp.date = start.date
	WHERE (tradingsymbol IN ($3))
),
new AS
(
	SELECT
		close,
		tradingsymbol
	FROM monkeybeat.prices AS sp
	INNER JOIN end ON sp.date = end.date
	WHERE (tradingsymbol IN ($3))
)
SELECT
new.tradingsymbol AS symbol,
((new.close / old.close) - 1) * 100 AS return_percent
FROM old
INNER JOIN new ON old.tradingsymbol = new.tradingsymbol

-- name: get-daily-value
-- Fetch the daily returns by computing the close price each day and fetching the percentage difference from starting date.
-- $1: stocks
-- $2: amount_invested
SELECT
formatDateTime(toDate(date), '%F') AS date,
SUM(close) AS present_close,
(
	SELECT SUM(close) AS close
	FROM monkeybeat.prices
	WHERE (tradingsymbol IN ($1))
	GROUP BY date
	ORDER BY date ASC
	LIMIT 1
) AS initial_close,
(100. * (present_close - initial_close)) / initial_close AS percent_diff,
$2 + ((percent_diff / 100) * $2) AS current_invested
FROM monkeybeat.prices
WHERE (tradingsymbol IN ($1))
GROUP BY date
ORDER BY date ASC
