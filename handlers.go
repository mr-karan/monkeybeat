package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

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

// wrap is a middleware that wraps HTTP handlers and injects the "app" context.
func wrap(app *App, next http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := context.WithValue(r.Context(), "app", app)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// resp is used to send uniform response structure.
type resp struct {
	Status  string      `json:"status"`
	Message string      `json:"message,omitempty"`
	Data    interface{} `json:"data,omitempty"`
}

// sendResponse sends a JSON envelope to the HTTP response.
func sendResponse(w http.ResponseWriter, code int, data interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)

	out, err := json.Marshal(resp{Status: "success", Data: data})
	if err != nil {
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	w.Write(out)
}

// sendErrorResponse sends a JSON error envelope to the HTTP response.
func sendErrorResponse(w http.ResponseWriter, message string, code int, data interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)

	resp := resp{
		Status:  "error",
		Message: message,
		Data:    data,
	}
	out, err := json.Marshal(resp)
	if err != nil {
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	w.Write(out)
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	var (
		app = r.Context().Value("app").(*App)
	)
	app.tpl.ExecuteTemplate(w, "base", viewTpl{
		ShowPortfolio: false,
	})
}

// Handler for generating a random portfolio.
func handlePortfolio(w http.ResponseWriter, r *http.Request) {
	var (
		app = r.Context().Value("app").(*App)
		out = ReturnResp{}
	)

	// Fetch a list of stocks from DB.
	stocks, err := app.getRandomStocks()
	if err != nil {
		app.lo.Error("error generating stocks", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	// Fetch portfolio returns for 6 months.
	out.Portfolio6M, err = app.getReturns(stocks, 30*6)
	if err != nil {
		app.lo.Error("error fetching returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	// Fetch portfolio returns for 1 year.
	out.Portfolio1Y, err = app.getReturns(stocks, 365)
	if err != nil {
		app.lo.Error("error fetching returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	// Fetch portfolio returns for 3 years.
	out.Portfolio3Y, err = app.getReturns(stocks, 365*3)
	if err != nil {
		app.lo.Error("error fetching returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	// Fetch Index returns for 6 months.
	index6M, err := app.getReturns([]string{N500_SYMBOL}, 30*6)
	if err != nil {
		app.lo.Error("error fetching returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	// Fetch Index returns for 1 year.
	index1Y, err := app.getReturns([]string{N500_SYMBOL}, 365)
	if err != nil {
		app.lo.Error("error fetching returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	// Fetch Index returns for 3 years.
	index3Y, err := app.getReturns([]string{N500_SYMBOL}, 365*3)
	if err != nil {
		app.lo.Error("error fetching returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	avgReturnsMap := make(map[string]AvgPortfolioReturns, len(out.Portfolio6M))
	for _, v := range out.Portfolio6M {
		avgReturnsMap[v.Symbol] = AvgPortfolioReturns{
			Avg6M: v.Percent,
		}
	}

	// for _,v:=range out.Portfolio6M {
	// 	avgReturnsMap[v.Symbol]=AvgPortfolioReturns{
	// 		Avg6M: v.Percent,
	// 	}
	// }

	// for _,v:=range out.Portfolio6M {
	// 	avgReturnsMap[v.Symbol]=AvgPortfolioReturns{
	// 		Avg6M: v.Percent,
	// 	}
	// }

	// Compute average returns.
	out.AvgPorfolio6M = computeAvg(out.Portfolio6M)
	out.AvgPorfolio1Y = computeAvg(out.Portfolio1Y)
	out.AvgPorfolio3Y = computeAvg(out.Portfolio3Y)
	out.AvgIndex6M = computeAvg(index6M)
	out.AvgIndex1Y = computeAvg(index1Y)
	out.AvgIndex3Y = computeAvg(index3Y)

	// Fetch the daily returns.
	dailyReturns, err := app.getDailyReturns(stocks, 10000)
	if err != nil {
		app.lo.Error("error fetching daily returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	fmt.Println(dailyReturns)

	app.tpl.ExecuteTemplate(w, "base", viewTpl{
		ShowPortfolio: true,
		ReturnResp:    out,
		DailyReturns:  dailyReturns,
	})
}

// Generate a random portfolio of 10 stocks.
func (app *App) getRandomStocks() ([]string, error) {
	stocks := make([]string, 10)
	if err := app.db.QueryRow(context.Background(), selectRandStocksSQL).Scan(&stocks); err != nil {
		return nil, err
	}
	return stocks, nil
}

// Fetch return percent for a given list of stocks and the given time period (in days).
func (app *App) getReturns(stocks []string, days int) ([]Returns, error) {
	returns := make([]Returns, 10)

	if err := app.db.Select(context.Background(), &returns, computeReturnsSQL, days, stocks); err != nil {
		return nil, err
	}
	return returns, nil
}

// Fetch current investment amount for each date since beginning to show overall amount.
func (app *App) getDailyReturns(stocks []string, amount int) ([]DailyReturns, error) {
	returns := make([]DailyReturns, 0)

	if err := app.db.Select(context.Background(), &returns, dailyReturnsSQL, stocks, amount); err != nil {
		return nil, err
	}
	return returns, nil
}

func computeAvg(ret []Returns) float64 {
	total := 0.0
	for _, r := range ret {
		total += r.Percent
	}
	return total / float64(len((ret)))
}
