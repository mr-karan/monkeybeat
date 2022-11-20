package main

import (
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
	"github.com/google/uuid"
	"github.com/zerodha/logf"
)

const (
	STOCKS_COUNT         = 10    // Number of stocks to add in a portfolio.
	NORMALIZATION_FACTOR = 100   // Factor to scale the inital price of each stock to calculate daily returns.
	PORTFOLIO_AMOUNT     = 10000 // Hypothetical starting amount invested in the portfolio.
)

var (
	// Time periods in days to calculate returns.
	// NOTE: If the values are changed here, they must be updated in HTML templates as well.
	returnPeriods = []int{30, 180, 365, 1095, 1825}
	indexSymbols  = []string{"NIFTY50", "NIFTY500", "NASDAQ100", "SP500"}
)

type App struct {
	lo      logf.Logger
	tpl     *template.Template
	db      driver.Conn
	queries *queries
}

type queries struct {
	GetRandomStocks string `query:"get-random-stocks"`
	GetReturns      string `query:"get-returns"`
	GetDailyValue   string `query:"get-daily-value"`
	InsertLink      string `query:"insert-link"`
	GetLink         string `query:"get-link"`
}

// Returns represent portfolio return for a single stock and a given timeframe.
type Returns struct {
	Symbol  string  `ch:"symbol" json:"symbol"`
	Percent float64 `ch:"return_percent" json:"percent"`
}

type LinkDetails struct {
	Portfolio string `ch:"portfolio"`
}

// ReturnsPeriod computes average returns for all stocks/indices for various time periods.
type ReturnsPeriod map[int][]Returns

// AvgStockReturns is a map of each stock with returns in different time periods.
type AvgStockReturns map[string]map[int]float64

// DailyReturns computes the returns for the entire portfolio for each date since beginning till current date.
type DailyReturns struct {
	Date            string  `ch:"close_date"`
	CurrentInvested float64 `ch:"current_invested"`
	NormalizedClose float64 `ch:"normalized_close"`
	ReturnPercent   float64 `ch:"return_percent"`
}

// getRandomStocks Generates a random portfolio of stocks for a given count and index catgeory.
func (app *App) getRandomStocks(count int, category string) ([]string, error) {
	stocks := make([]string, 0)
	if err := app.db.QueryRow(context.Background(), app.queries.GetRandomStocks, count, category).Scan(&stocks); err != nil {
		return nil, err
	}
	return stocks, nil
}

// getPortfolioReturns fetches return percent for a given list of stocks and the given time period (in days).
func (app *App) getPortfolioReturns(stocks []string, days int) ([]Returns, error) {
	return app.getReturns(days, stocks)
}

// getIndexReturns fetches return percent for a given list of indices and the given time period (in days).
func (app *App) getIndexReturns(indices []string, days int) ([]Returns, error) {
	return app.getReturns(days, indices)
}

func (app *App) getReturns(days int, stocks []string) ([]Returns, error) {
	returns := make([]Returns, 0)
	if err := app.db.Select(context.Background(), &returns, app.queries.GetReturns, days, stocks); err != nil {
		return nil, err
	}
	return returns, nil
}

// Fetch current investment amount for each date since beginning to show overall amount.
func (app *App) getDailyValue(stocks []string, days int) ([]DailyReturns, error) {
	returns := make([]DailyReturns, 0)
	if err := app.db.Select(context.Background(), &returns, app.queries.GetDailyValue, stocks, PORTFOLIO_AMOUNT, days, NORMALIZATION_FACTOR); err != nil {
		return nil, err
	}
	return returns, nil
}

// savePortfolio generates a unique UUID for the given portfolio and saves it to
// share table.
func (app *App) savePortfolio(data portfolioTpl) (string, error) {
	uuid := uuid.New()

	// Save the portfolio data as a JSON string.
	pf, err := json.Marshal(data)
	if err != nil {
		return "", err
	}

	batch, err := app.db.PrepareBatch(context.Background(), app.queries.InsertLink)
	if err != nil {
		return "", fmt.Errorf("error preparing batch: %v", err)
	}
	if err = batch.Append(time.Now(), uuid, pf); err != nil {
		return "", fmt.Errorf("error appending data to batch: %v", err)
	}
	if err = batch.Send(); err != nil {
		return "", fmt.Errorf("error inserting data: %v", err)
	}

	return uuid.String(), nil
}

// Fetch the portfolio data for a given UUID.
func (app *App) getLink(uuid string) (portfolioTpl, error) {
	var (
		out portfolioTpl
	)
	data := make([]LinkDetails, 0)

	if err := app.db.Select(context.Background(), &data, app.queries.GetLink, uuid); err != nil {
		return out, fmt.Errorf("error fetching uuid: %v", err)
	}

	if len(data) == 0 {
		return out, fmt.Errorf("no records fetched for given uuid")
	}

	if err := json.Unmarshal([]byte(data[0].Portfolio), &out); err != nil {
		return out, fmt.Errorf("error unmarshalling data: %v", err)
	}

	return out, nil
}

// computeAvg iterates on the change percent in Returns and computes an average.
func computeAvg(ret []Returns) float64 {
	total := 0.0
	if len(ret) == 0 {
		return total
	}

	for _, r := range ret {
		total += r.Percent
	}
	return total / float64(len((ret)))
}

// validIndex loops over a list of valid index symbols and returns false if it's an unknown symbol.
func validIndex(i string) bool {
	for _, idx := range indexSymbols {
		if idx == i {
			return true
		}
	}
	return false
}
