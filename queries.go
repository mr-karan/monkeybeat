package main

var computeReturnsSQL = `WITH
start AS (
	SELECT date FROM stocks.prices WHERE toDate(date)>=today() - INTERVAL $1 DAY ORDER BY date ASC LIMIT 1
),
end AS (
	SELECT date FROM stocks.prices WHERE toDate(date)>=today() - INTERVAL $1 DAY ORDER BY date DESC LIMIT 1
),
old AS
(
	SELECT
		close,
		tradingsymbol
	FROM stocks.prices AS sp
	INNER JOIN start ON sp.date = start.date
	WHERE (tradingsymbol IN ($2))
),
new AS
(
	SELECT
		close,
		tradingsymbol
	FROM stocks.prices AS sp
	INNER JOIN end ON sp.date = end.date
	WHERE (tradingsymbol IN ($2))
)
SELECT
new.tradingsymbol AS symbol,
((new.close / old.close) - 1) * 100 AS return_percent
FROM old
INNER JOIN new ON old.tradingsymbol = new.tradingsymbol
`

var selectRandStocksSQL = `SELECT groupArraySample(10)(tradingsymbol) AS stocks FROM stocks.prices`

var dailyReturnsSQL = `SELECT
formatDateTime(toDate(date), '%F') AS date,
SUM(close) AS present_close,
(
	SELECT SUM(close) AS close
	FROM stocks.prices
	WHERE (tradingsymbol IN ($1))
	GROUP BY date
	ORDER BY date ASC
	LIMIT 1
) AS initial_close,
(100. * (present_close - initial_close)) / initial_close AS percent_diff,
$2 + ((percent_diff / 100) * $2) AS current_invested
FROM stocks.prices
WHERE (tradingsymbol IN ($1))
GROUP BY date
ORDER BY date ASC
`
