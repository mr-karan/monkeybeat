package main

import (
	"context"
	"fmt"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
	"github.com/zerodha/logf"

	"os"
	"strings"

	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/knadh/koanf"
	"github.com/knadh/koanf/parsers/toml"
	"github.com/knadh/koanf/providers/env"
	"github.com/knadh/koanf/providers/file"
	flag "github.com/spf13/pflag"
)

// initConfig loads config to `ko` object.
func initConfig() (*koanf.Koanf, error) {
	var (
		ko = koanf.New(".")
		f  = flag.NewFlagSet("front", flag.ContinueOnError)
	)

	// Configure Flags.
	f.Usage = func() {
		fmt.Println(f.FlagUsages())
		os.Exit(0)
	}

	// Register `--config` flag.
	cfgPath := f.String("config", "config.sample.toml", "Path to a config file to load.")

	// Parse and Load Flags.
	err := f.Parse(os.Args[1:])
	if err != nil {
		return nil, err
	}

	err = ko.Load(file.Provider(*cfgPath), toml.Parser())
	if err != nil {
		return nil, err
	}
	err = ko.Load(env.Provider("MONKEYBEAT_", ".", func(s string) string {
		return strings.Replace(strings.ToLower(
			strings.TrimPrefix(s, "MONKEYBEAT_")), "__", ".", -1)
	}), nil)
	if err != nil {
		return nil, err
	}
	return ko, nil
}

// initClickhouse initialises clickhouse conncetion using the native interface.
func initClickhouse(ko *koanf.Koanf) (driver.Conn, error) {
	conn, err := clickhouse.Open(&clickhouse.Options{
		Addr:        []string{fmt.Sprintf("%s:%d", ko.String("db.host"), ko.Int("db.port"))},
		DialTimeout: time.Second * 5,
		ReadTimeout: time.Minute * 1,
		Debug:       false,
		Auth: clickhouse.Auth{
			Database: ko.String("db.name"),
			Username: ko.String("db.user"),
			Password: ko.String("db.password"),
		},
	})
	if err != nil {
		return nil, err
	}

	if err := conn.Ping(context.Background()); err != nil {
		return nil, err
	}

	return conn, nil
}

// initLogger initializes logger instance.
func initLogger(ko *koanf.Koanf) logf.Logger {
	opts := logf.Opts{EnableCaller: true, EnableColor: true}
	if ko.String("app.log") == "debug" {
		opts.Level = logf.DebugLevel
	}
	return logf.New(opts)
}
