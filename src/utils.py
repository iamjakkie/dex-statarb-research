# load s3
# fetch token data (if possible)

import tokens

import polars as pl

SCHEMAS = {
    "DYDX": pl.Schema({
        "startedAt": pl.Datetime("ms", None),
        "ticker": pl.String,
        "resolution": pl.String,
        "low": pl.Float64,
        "high": pl.Float64,
        "open": pl.Float64,
        "close": pl.Float64,
        "baseTokenVolume": pl.Float64,
        "usdVolume": pl.Float64,
        "trades": pl.Float64,
        "startingOpenInterest": pl.Float64,
        "orderbookMidPriceOpen": pl.Float64,
        "orderbookMidPriceClose": pl.Float64,
    }),

    "HYPERLIQUID": pl.Schema({
        "time": pl.Datetime("ms", None),
        "coin": pl.String,
        "funding": pl.Float64,
        "open_interest": pl.Float64,
        "prev_day_px": pl.Float64,
        "day_ntl_vim": pl.Float64,
        "premium": pl.Float64,
        "oracle_px": pl.Float64,
        "mark_px": pl.Float64,
        "mid_px": pl.Float64,
        "impact_bid_px": pl.Float64,
        "impact_ask_px": pl.Float64,
    })
}

def load_data(src: str, token: str) -> pl.LazyFrame:

    if src == "SOLANA":
        token_clean = tokens.TOKEN_MAPPING[token]['sol']
        if not token_clean:
            raise ValueError(f"Token {token} not found in tokens.TOKEN_MAPPING for SOLANA")
        s3_path = f"s3://iamjakkie-public/normalized/solana_swaps/*/PROGRAM_ID=*/TOKEN={token_clean}/QUOTE_ASSET=*/*.parquet"
    elif src == "ETHEREUM":
        token_clean = tokens.TOKEN_MAPPING[token]['eth']
        if not token_clean:
            raise ValueError(f"Token {token} not found in tokens.TOKEN_MAPPING for ETHEREUM")
        s3_path = f"s3://iamjakkie-public/normalized/ethereum_swaps/*/PLATFORM=*/TOKEN={token_clean}/*.parquet"
    elif src == "BASE":
        token_clean = tokens.TOKEN_MAPPING[token]['base']
        if not token_clean:
            raise ValueError(f"Token {token} not found in tokens.TOKEN_MAPPING for BASE")
        s3_path = f"s3://iamjakkie-public/normalized/base_swaps/*/PLATFORM=*/TOKEN={token_clean}/*.parquet"
    elif src == "DYDX":
        # Parse cols, combine with funding rates
        token_clean = tokens.TOKEN_MAPPING[token]['dydx']
        if not token_clean:
            raise ValueError(f"Token {token} not found in tokens.TOKEN_MAPPING for DYDX")
        s3_path = f"s3://iamjakkie-public/dydx/candles/{token_clean}.parquet"
    elif src == "HYPERLIQUID":
        token_clean = tokens.TOKEN_MAPPING[token]['hyperliquid']
        if not token_clean:
            raise ValueError(f"Token {token} not found in tokens.TOKEN_MAPPING for HYPERLIQUID")
        s3_path = f"s3://iamjakkie-public/hyperliquid/*/coin={token_clean}/*.parquet"
    elif src == "DRIFT":
        token_clean = tokens.TOKEN_MAPPING[token]['drift']
        if not token_clean:
            raise ValueError(f"Token {token} not found in tokens.TOKEN_MAPPING for DRIFT")
        s3_path = f"s3://iamjakkie-public/drift/{token_clean}/trades/*.parquet"
    else:
        raise ValueError("src must be one of SOLANA, ETHEREUM, BASE, DYDX, HYPERLIQUID, DRIFT")

    print(s3_path)

    df = pl.scan_parquet(s3_path)
    df = clean_data(df, src, token)

    return df


def clean_data(df: pl.DataFrame, src: str, token: str) -> pl.DataFrame:
    """
    Cap the price of the token to its max price
    """
    if token not in tokens.TOKEN_MAPPING:
        raise ValueError(f"Token {token} not found in tokens.TOKEN_MAPPING")

    if src == "SOLANA":
        return price_roof(df, token, "USD_PRICE")

    elif src == "DYDX":
        clean_by_schema(df, src)
        return price_roof(df, token, "close")
    
    elif src == "HYPERLIQUID":
        clean_by_schema(df, src)
        return price_roof(df, token, "mid_px")
    
def clean_by_schema(df: pl.LazyFrame, src: str) -> pl.LazyFrame:
    exprs = []
    for col, dtype in SCHEMAS[src].items():
        if isinstance(dtype, pl.Datetime):
            exprs.append(
                pl.col(col)
                .str.strptime(dtype, "%Y-%m-%dT%H:%M:%S%.3fZ")
                .alias(col)
            )
        else:
            exprs.append(
                pl.col(col).cast(dtype).alias(col)
            )
    df.with_columns(exprs)

def price_roof(df: pl.LazyFrame, token: str,  price_col: str) -> pl.Expr:
    max_price = tokens.TOKEN_MAPPING[token]['max_price']
    return (df
                .filter(pl.col(price_col).is_finite())
                .with_columns(
                    pl.when(pl.col(price_col) > max_price)
                        .then(max_price)
                        .otherwise(pl.col(price_col))
                        .alias(price_col)
                ))