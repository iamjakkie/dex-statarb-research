# load s3
# fetch token data (if possible)

import tokens

import polars as pl

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
    df = roof_price(df, src, token)

    return df


def roof_price(df: pl.DataFrame, src: str, token: str) -> pl.DataFrame:
    """
    Cap the price of the token to its max price
    """
    if token not in tokens.TOKEN_MAPPING:
        raise ValueError(f"Token {token} not found in tokens.TOKEN_MAPPING")
    
    print(tokens.TOKEN_MAPPING[token])
    max_price = tokens.TOKEN_MAPPING[token]['max_price']
    if max_price is None:
        raise ValueError(f"Max price for token {token} not found in tokens.TOKEN_MAPPING")

    if src == "SOLANA":
        return (df
                .filter(pl.col("USD_PRICE").is_finite())
                .with_columns(
                    pl.when(pl.col("USD_PRICE") > max_price)
                    .then(max_price)
                    .otherwise(pl.col("USD_PRICE"))
                    .alias("USD_PRICE")
                ))