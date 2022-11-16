package main

import (
	"context"
	"text/template"

	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"

	"github.com/zerodha/logf"
)

const (
	STOCKS_COUNT = 10
)

type App struct {
	lo      logf.Logger
	tpl     *template.Template
	db      driver.Conn
	queries *queries
}

// Queries contains all prepared SQL queries.
type queries struct {
	GetRandomStocks string `query:"get-random-stocks"`
	GetReturns      string `query:"get-returns"`
	GetDailyReturns string `query:"get-daily-returns"`
}

const N500_SYMBOL = "^CRSLDX"

type Returns struct {
	Symbol  string  `ch:"symbol" json:"symbol"`
	Percent float64 `ch:"return_percent" json:"percent"`
}

type DailyReturns struct {
	Date            string  `ch:"date"`
	CurrentInvested float64 `ch:"current_invested"`
	PresentClose    float64 `ch:"present_close"`
	InitalClose     float64 `ch:"initial_close"`
	PercentDiff     float64 `ch:"percent_diff"`
}

type ReturnResp struct {
	Portfolio6M []Returns `json:"port_6m"`
	Portfolio1Y []Returns `json:"port_1y"`
	Portfolio3Y []Returns `json:"port_3y"`

	AvgPorfolio6M float64 `json:"avg_port_6m"`
	AvgPorfolio1Y float64 `json:"avg_port_1y"`
	AvgPorfolio3Y float64 `json:"avg_port_3y"`
	AvgIndex6M    float64 `json:"avg_ind_6m"`
	AvgIndex1Y    float64 `json:"avg_ind_1y"`
	AvgIndex3Y    float64 `json:"avg_ind_3y"`
}

type AvgPortfolioReturns struct {
	Avg6M float64
	Avg1Y float64
	Avg3Y float64
}

type viewTpl struct {
	ShowPortfolio bool
	ReturnResp
	DailyReturns []DailyReturns
}

// Generate a random portfolio of 10 stocks.
func (app *App) getRandomStocks() ([]string, error) {
	stocks := make([]string, STOCKS_COUNT)
	if err := app.db.QueryRow(context.Background(), app.queries.GetRandomStocks, STOCKS_COUNT).Scan(&stocks); err != nil {
		return nil, err
	}
	return stocks, nil
}

// Fetch return percent for a given list of stocks and the given time period (in days).
func (app *App) getReturns(stocks []string, days int) ([]Returns, error) {
	returns := make([]Returns, STOCKS_COUNT)
	if err := app.db.Select(context.Background(), &returns, app.queries.GetReturns, days, stocks); err != nil {
		return nil, err
	}
	return returns, nil
}

// Fetch current investment amount for each date since beginning to show overall amount.
func (app *App) getDailyReturns(stocks []string, amount int) ([]DailyReturns, error) {
	returns := make([]DailyReturns, 0)
	if err := app.db.Select(context.Background(), &returns, app.queries.GetDailyReturns, stocks, amount); err != nil {
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
