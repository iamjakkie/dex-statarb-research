# load s3
# fetch token data (if possible)

from tokens import TOKEN_MAPPING

import polars as pl

def load_data(src: str, token: str) -> pl.LazyFrame:

    if src == "SOLANA":
        token_clean = TOKEN_MAPPING[token]['sol']
        if not token_clean:
            raise ValueError(f"Token {token} not found in TOKEN_MAPPING for SOLANA")
        s3_path = f"s3://iamjakkie-public/normalized/solana_swaps/*/PROGRAM_ID=*/TOKEN={token_clean}/QUOTE_ASSET=*/*.parquet"
    elif src == "ETHEREUM":
        token_clean = TOKEN_MAPPING[token]['eth']
        if not token_clean:
            raise ValueError(f"Token {token} not found in TOKEN_MAPPING for ETHEREUM")
        s3_path = f"s3://iamjakkie-public/normalized/ethereum_swaps/*/PLATFORM=*/TOKEN={token_clean}/*.parquet"
    elif src == "BASE":
        token_clean = TOKEN_MAPPING[token]['base']
        if not token_clean:
            raise ValueError(f"Token {token} not found in TOKEN_MAPPING for BASE")
        s3_path = f"s3://iamjakkie-public/normalized/base_swaps/*/PLATFORM=*/TOKEN={token_clean}/*.parquet"
    elif src == "DYDX":
        token_clean = TOKEN_MAPPING[token]['dydx']
        if not token_clean:
            raise ValueError(f"Token {token} not found in TOKEN_MAPPING for DYDX")
        s3_path = f"s3://iamjakkie-public/dydx/candles/{token_clean}.parquet"
    elif src == "HYPERLIQUID":
        token_clean = TOKEN_MAPPING[token]['hyperliquid']
        if not token_clean:
            raise ValueError(f"Token {token} not found in TOKEN_MAPPING for HYPERLIQUID")
        s3_path = f"s3://iamjakkie-public/hyperliquid/*/coin={token_clean}/*.parquet"
    elif src == "DRIFT":
        token_clean = TOKEN_MAPPING[token]['drift']
        if not token_clean:
            raise ValueError(f"Token {token} not found in TOKEN_MAPPING for DRIFT")
        s3_path = f"s3://iamjakkie-public/drift/{token_clean}/trades/*.parquet"
    else:
        raise ValueError("src must be one of SOLANA, ETHEREUM, BASE, DYDX, HYPERLIQUID, DRIFT")

    print(s3_path)
    return pl.scan_parquet(s3_path)
