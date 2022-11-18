-- monkeybeat
-- name: get-random-stocks
-- Fetch a list of random stocks for the given count.
-- $1: count
SELECT groupArraySample($1)(tradingsymbol) AS stocks FROM monkeybeat.prices WHERE segment='EQ'

-- name: get-returns
-- Get average returns for given date and given list of stocks.
-- $1: date
-- $2: symbols
WITH
start AS (
	SELECT date FROM monkeybeat.prices WHERE toDate(date)>=today() - INTERVAL $1 DAY ORDER BY date ASC LIMIT 1
),
end AS (
	SELECT date FROM monkeybeat.prices WHERE toDate(date)>=today() - INTERVAL $1 DAY ORDER BY date DESC LIMIT 1
),
old AS
(
	SELECT
		close,
		tradingsymbol
	FROM monkeybeat.prices AS sp
	INNER JOIN start ON sp.date = start.date
	WHERE (tradingsymbol IN ($2))
),
new AS
(
	SELECT
		close,
		tradingsymbol
	FROM monkeybeat.prices AS sp
	INNER JOIN end ON sp.date = end.date
	WHERE (tradingsymbol IN ($2))
)
SELECT
new.tradingsymbol AS symbol,
((new.close / old.close) - 1) * 100 AS return_percent
FROM old
INNER JOIN new ON old.tradingsymbol = new.tradingsymbol

-- name: get-daily-value
-- Fetch the daily returns by computing the close price each day and fetching the percentage difference from starting date.
-- The starting amount of each stock is set as $4 (normalization_factor). The closing price each day is computed by factoring in $4.
-- $1: stocks
-- $2: amount_invested
-- $3: days
-- $4: normalization_factor
WITH
    initial AS
    (
        SELECT
            tradingsymbol,
            $4 / close AS multiplier,
            close
        FROM monkeybeat.prices
        WHERE (tradingsymbol IN ($1)) AND (date = (
            SELECT date
            FROM monkeybeat.prices
            WHERE date >= (today() - toIntervalDay($3))
            ORDER BY date ASC
            LIMIT 1
        ))
    ),
    present AS
    (
        SELECT
            date,
            tradingsymbol,
            close
        FROM monkeybeat.prices
        WHERE (tradingsymbol IN ($1)) AND (date >= (today() - toIntervalDay($3)))
        ORDER BY date ASC
    )
SELECT
    formatDateTime(toDate(present.date), '%F') AS close_date,
    SUM(initial.multiplier * present.close) AS normalized_close,
    ((normalized_close / ($4 * count(present.tradingsymbol))) - 1) * 100 AS return_percent,
	$2 + ((return_percent / 100) *10000) AS current_invested
FROM initial
INNER JOIN present ON initial.tradingsymbol = present.tradingsymbol
GROUP BY close_date
ORDER BY close_date ASC

-- name: insert-link
--- Insert into link table.
INSERT INTO monkeybeat.link

-- name: get-link
--- Fetch data from the links table for a given UUID.
-- $1: uuid
SELECT portfolio
FROM link
WHERE uuid=$1
