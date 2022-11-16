package main

import (
	"context"
	"text/template"

	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"

	"github.com/zerodha/logf"
)

const (
	STOCKS_COUNT = 10        // Number of stocks to add in a portfolio.
	N500_SYMBOL  = "^CRSLDX" // Index symbol.
)

var (
	returnPeriods = []int{30, 180, 360, 1080}
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
}

// Returns represent portfolio return for a single stock and a given timeframe.
type Returns struct {
	Symbol  string  `ch:"symbol" json:"symbol"`
	Percent float64 `ch:"return_percent" json:"percent"`
}

// DailyReturns computes the returns for the entire portfolio for each date since beginning till current date.
type DailyReturns struct {
	Date            string  `ch:"date"`
	CurrentInvested float64 `ch:"current_invested"`
	PresentClose    float64 `ch:"present_close"`
	InitalClose     float64 `ch:"initial_close"`
	PercentDiff     float64 `ch:"percent_diff"`
}

// ReturnsPeriod computes average returns for all stocks/indices for various time periods.
type ReturnsPeriod map[int][]Returns

// AvgReturns computes average returns for portfolio or index for various time periods.
type AvgReturns struct {
	AvgPorfolio1M float64 `json:"avg_port_1m"`
	AvgPorfolio3M float64 `json:"avg_port_3m"`
	AvgPorfolio6M float64 `json:"avg_port_6m"`
	AvgPorfolio1Y float64 `json:"avg_port_1y"`
	AvgPorfolio3Y float64 `json:"avg_port_3y"`

	AvgIndex1M float64 `json:"avg_ind_1m"`
	AvgIndex3M float64 `json:"avg_ind_3m"`
	AvgIndex6M float64 `json:"avg_ind_6m"`
	AvgIndex1Y float64 `json:"avg_ind_1y"`
	AvgIndex3Y float64 `json:"avg_ind_3y"`
}

// getRandomStocks Generates a random portfolio of stocks for a given count.
func (app *App) getRandomStocks(count int) ([]string, error) {
	stocks := make([]string, 0)
	if err := app.db.QueryRow(context.Background(), app.queries.GetRandomStocks, count).Scan(&stocks); err != nil {
		return nil, err
	}
	return stocks, nil
}

// getPortfolioReturns fetches return percent for a given list of stocks and the given time period (in days).
func (app *App) getPortfolioReturns(stocks []string, days int) ([]Returns, error) {
	return app.getReturns("stocks", days, stocks)
}

// getIndexReturns fetches return percent for a given list of indices and the given time period (in days).
func (app *App) getIndexReturns(indices []string, days int) ([]Returns, error) {
	return app.getReturns("index", days, indices)
}

func (app *App) getReturns(table string, days int, stocks []string) ([]Returns, error) {
	returns := make([]Returns, 0)
	if err := app.db.Select(context.Background(), &returns, app.queries.GetReturns, table, days, stocks); err != nil {
		return nil, err
	}
	return returns, nil

}

// Fetch current investment amount for each date since beginning to show overall amount.
func (app *App) getDailyValue(stocks []string, amount int) ([]DailyReturns, error) {
	returns := make([]DailyReturns, 0)
	if err := app.db.Select(context.Background(), &returns, app.queries.GetDailyValue, stocks, amount); err != nil {
		return nil, err
	}
	return returns, nil
}

// computeAvg iterates on the change percent in Returns and computes an average.
func computeAvg(ret []Returns) float64 {
	total := 0.0
	for _, r := range ret {
		total += r.Percent
	}
	return total / float64(len((ret)))
}