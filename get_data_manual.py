from binance_historical_data import BinanceDataDumper
import datetime

data_dumper = BinanceDataDumper(
    path_dir_where_to_dump="/Users/elizabethpaint/code/egppp/cryptoteller/data",
    asset_class="spot",  # spot, um, cm
    data_type="klines",  # aggTrades, klines, trades
    data_frequency="1d",
)

tickers=["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

data_dumper.dump_data(
    tickers=tickers,
    date_start=datetime.date(year=2022, month=4, day=30),
    date_end=datetime.date(year=2022, month=10, day=31),
    is_to_update_existing=False,
    tickers_to_exclude=["UST"],
)