package main

import (
	"embed"
	"fmt"
	"html/template"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

var (
	// Version and date of the build. This is injected at build-time.
	buildString = "unknown"
	//go:embed static/*
	staticDir embed.FS
	//go:embed queries.sql
	sqlFiles embed.FS
)

func main() {
	// Initialise and load the config.
	ko, err := initConfig()
	if err != nil {
		fmt.Printf("error loading config: %v", err)
		os.Exit(-1)
	}

	app := &App{
		lo: initLogger(ko),
	}
	app.lo.Info("booting monkeybeat", "version", buildString)

	// Initialise clickhouse DB connection.
	db, err := initClickhouse(ko)
	if err != nil {
		app.lo.Fatal("couldn't initialise clickhouse connection", "error", err)
	}
	app.db = db
	defer app.db.Close()

	queries, err := initQueries(sqlFiles)
	if err != nil {
		app.lo.Fatal("couldn't initialise queries", "error", err)
	}
	app.queries = queries

	// Load HTML templates.
	funcMap := template.FuncMap{
		"FormatNumber": func(value float64) string {
			return fmt.Sprintf("%.2f%"+"%", value)
		},
		"Color": func(val string) string {
			color := "green"
			if strings.HasPrefix(val, "-") {
				color = "red"
			}
			return fmt.Sprintf("<span class=%s>%s</span>", color, val)
		},
		"SafeHTML": func(val string) template.HTML {
			return template.HTML(val)
		},
		"TwitterShare": func(id string) string {
			link, _ := url.JoinPath(ko.MustString("app.domain"), "/portfolio", id)
			twtURL := "https://twitter.com/intent/tweet?text="
			txt := fmt.Sprintf("Check out my awesome portfolio on Monkeybeat. Visit %s via #monkeybeat", link)
			return twtURL + txt
		},
		"WhatsappShare": func(id string) string {
			link, _ := url.JoinPath(ko.MustString("app.domain"), "/portfolio", id)
			waURL := "https://api.whatsapp.com/send/?text="
			txt := fmt.Sprintf("Check out my awesome portfolio on Monkeybeat. Visit %s", link)
			return waURL + txt
		},
	}
	tpl, err := template.New("static").Funcs(funcMap).ParseFS(staticDir, "static/*.html")
	if err != nil {
		app.lo.Fatal("couldn't load html templates", "error", err)
	}
	app.tpl = tpl

	// Register router instance.
	r := chi.NewRouter()
	// Register middlewares
	r.Use(middleware.Logger)

	// Frontend Handlers.
	r.Get("/", wrap(app, handleIndex))
	r.Get("/portfolio", wrap(app, handlePortfolio))
	r.Get("/portfolio/{uuid}", wrap(app, handlePortfolio))

	// Static assets.
	static, _ := fs.Sub(staticDir, "static")
	r.Get("/static/*", func(w http.ResponseWriter, r *http.Request) {
		fs := http.StripPrefix("/static/", http.FileServer(http.FS(static)))
		fs.ServeHTTP(w, r)
	})

	// HTTP Server.
	srv := &http.Server{
		Addr:         ko.String("server.address"),
		Handler:      r,
		ReadTimeout:  ko.Duration("server.timeout") * time.Millisecond,
		WriteTimeout: ko.Duration("server.timeout") * time.Millisecond,
		IdleTimeout:  ko.Duration("server.idle_timeout") * time.Millisecond,
	}

	app.lo.Info("starting server", "address", srv.Addr)
	if err := srv.ListenAndServe(); err != nil {
		app.lo.Fatal("couldn't start server", "error", err)
	}
}
