package main

import (
	"context"
	"encoding/json"
	"net/http"
)

// portfolioTpl is used to represent the data which is used to render
// portfolio web view.
type portfolioTpl struct {
	DailyPortfolioReturns  []DailyReturns
	DailyIndexReturns      []DailyReturns
	AvgStockReturns        AvgStockReturns
	AvgIndexReturns        map[int]float64
	AvgPortfolioReturns    map[int]float64
	CurrentPortfolioAmount int64
	CurrentIndexAmount     int64
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
	// TODO: Have an error.html?
	out, err := json.Marshal(resp)
	if err != nil {
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	w.Write(out)
}

// handleIndex serves the index page.
func handleIndex(w http.ResponseWriter, r *http.Request) {
	var (
		app = r.Context().Value("app").(*App)
	)
	app.tpl.ExecuteTemplate(w, "index", nil)
}

// handlePortfolio serves the portfolio page.
func handlePortfolio(w http.ResponseWriter, r *http.Request) {
	var (
		app                   = r.Context().Value("app").(*App)
		portfolio             = make(ReturnsPeriod, 0)
		index                 = make(ReturnsPeriod, 0)
		avgStockReturns       = make(AvgStockReturns, 0)
		avgPortfolioReturns   = make(map[int]float64, 0)
		avgIndexReturns       = make(map[int]float64, 0)
		dailyPortfolioReturns = make([]DailyReturns, 0)
		dailyIndexReturns     = make([]DailyReturns, 0)
	)

	// Fetch a list of stocks from DB.
	stocks, err := app.getRandomStocks(STOCKS_COUNT)
	if err != nil {
		app.lo.Error("error generating stocks", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	// Fetch returns for various time periods.
	for _, days := range returnPeriods {
		returns, err := app.getPortfolioReturns(stocks, days)
		if err != nil {
			app.lo.Error("error fetching portfolio returns", "error", err)
			sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
			return
		}

		portfolio[days] = returns
		avgPortfolioReturns[days] = computeAvg(returns)

		// Add individual stock returns.
		for _, s := range returns {
			if val, ok := avgStockReturns[s.Symbol]; ok {
				val[days] = s.Percent
			} else {
				avgStockReturns[s.Symbol] = map[int]float64{days: s.Percent}
			}
		}
	}

	// Fetch index returns for various time periods.
	for _, days := range returnPeriods {
		returns, err := app.getIndexReturns([]string{N500_SYMBOL}, days)
		if err != nil {
			app.lo.Error("error fetching index returns", "error", err)
			sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
			return
		}
		index[days] = returns
		avgIndexReturns[days] = computeAvg(returns)
	}

	// Fetch the daily returns over 3 years.
	dailyPortfolioReturns, err = app.getDailyValue(stocks, 1080)
	if err != nil {
		app.lo.Error("error fetching daily returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}
	dailyIndexReturns, err = app.getDailyValue([]string{N500_SYMBOL}, 1080)
	if err != nil {
		app.lo.Error("error fetching daily returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	if len(dailyIndexReturns) == 0 || len(dailyPortfolioReturns) == 0 {
		app.lo.Error("error fetching daily returns", "error", err)
		sendErrorResponse(w, "Internal Server Error.", http.StatusInternalServerError, nil)
		return
	}

	app.tpl.ExecuteTemplate(w, "portfolio", portfolioTpl{
		DailyPortfolioReturns:  dailyPortfolioReturns,
		DailyIndexReturns:      dailyIndexReturns,
		AvgStockReturns:        avgStockReturns,
		AvgIndexReturns:        avgIndexReturns,
		AvgPortfolioReturns:    avgPortfolioReturns,
		CurrentPortfolioAmount: int64(dailyPortfolioReturns[len(dailyPortfolioReturns)-1].CurrentInvested),
		CurrentIndexAmount:     int64(dailyIndexReturns[len(dailyIndexReturns)-1].CurrentInvested),
	})
}
