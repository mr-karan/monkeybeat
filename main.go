package main

import (
	"embed"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"text/template"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/zerodha/logf"
)

var (
	// Version and date of the build. This is injected at build-time.
	buildString = "unknown"
	//go:embed assets/*
	assetsDir embed.FS
)

type App struct {
	lo  logf.Logger
	tpl *template.Template
	db  driver.Conn
}

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

	// Initialise clickhouse.
	db, err := initClickhouse(ko)
	if err != nil {
		app.lo.Fatal("couldn't initialise clickhouse connection", "error", err)
	}
	app.db = db

	// Load HTML templates.
	tpl, err := template.ParseFS(assetsDir, "assets/*.html")
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
	assets, _ := fs.Sub(assetsDir, "assets")
	r.Get("/assets/*", func(w http.ResponseWriter, r *http.Request) {
		fs := http.StripPrefix("/assets/", http.FileServer(http.FS(assets)))
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
