#!/usr/bin/env python
import sys
import asyncio
import logging
import time
import pandas as pd
from openbb import obb

# On Windows, use the Selector event loop to avoid "Event loop is closed" warnings.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Define alternate indicator codes for IMF (adjust as needed)
imf_alternate = {
    "PCPIPCH": "PCPI_CH",   # Alternate for inflation
    "NGDP_RPCH": "NGDP_CH"    # Alternate for growth
}

def print_available_indicators(provider):
    """
    Print available indicators for the given provider.
    """
    try:
        available = obb.economy.available_indicators(provider=provider)
        print(f"\nAvailable indicators for provider '{provider}':")
        print(available)
    except Exception as e:
        print(f"Error retrieving available indicators for provider '{provider}': {e}")

def clean_imf_dataframe(df):
    """
    IMF response data often comes as a DataFrame with columns containing tuple values,
    e.g. (key, value). This function converts each row into a dictionary by extracting
    the second element of each tuple.
    """
    cleaned_rows = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            val = row[col]
            # If the cell is a tuple of the form (key, value), use value
            if isinstance(val, tuple) and len(val) == 2:
                row_dict[col] = val[1]
            else:
                row_dict[col] = val
        cleaned_rows.append(row_dict)
    return pd.DataFrame(cleaned_rows)

def fetch_data(provider, symbol, country, start_date, end_date, max_retries=3):
    """
    Fetch data using obb.economy.indicators with the specified parameters.
    If provider is IMF and no data is found, automatically try an alternate indicator code.
    """
    def attempt_fetch(sym):
        for attempt in range(1, max_retries + 1):
            logging.info(f"[{provider.upper()}] Attempt {attempt}/{max_retries} for {sym}")
            try:
                data = obb.economy.indicators(
                    provider=provider,
                    symbol=sym,
                    country=country,
                    start_date=start_date,
                    end_date=end_date
                )
                if hasattr(data, "results") and data.results:
                    df = pd.DataFrame(data.results)
                    # Clean tuple values if needed (common with IMF data)
                    if df.shape[1] >= 4 and isinstance(df.iloc[0, 0], tuple):
                        df = clean_imf_dataframe(df)
                    logging.info(f"[{provider.upper()}] Successfully fetched data for {sym}")
                    return df
                else:
                    logging.error(f"[{provider.upper()}] Attempt {attempt} returned no time series data for {sym}")
            except Exception as e:
                logging.error(f"[{provider.upper()}] Attempt {attempt} failed for {sym}: {e}")
            time.sleep(2 * attempt)  # Increasing delay between attempts
        return pd.DataFrame()

    # First try the primary indicator code
    df = attempt_fetch(symbol)
    # If no data was returned and provider is IMF, try an alternate code if available.
    if provider.lower() == "imf" and df.empty and symbol in imf_alternate:
        alt_symbol = imf_alternate[symbol]
        logging.info(f"[IMF] No data for {symbol}. Trying alternate indicator code: {alt_symbol}")
        df = attempt_fetch(alt_symbol)
    if df.empty:
        logging.error(f"[{provider.upper()}] Failed to fetch data for {symbol}. Verify the indicator code, country, and date range.")
    return df

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Define a wide date range to ensure data exists
    start_date = "1980-01-01"
    end_date = "2023-01-01"
    
    # For ECONDB, use full lowercase country names with underscores.
    # For IMF, you can often use ISO codes.
    country_econdb = "united_states"
    country_imf = "us"
    
    # Print available indicators from both providers (helps to verify current symbols)
    print_available_indicators("econdb")
    print_available_indicators("imf")
    
    # Define which symbols to use for each indicator.
    # Adjust these symbols based on the output from available_indicators.
    indicators = {
        "Inflation": {"econdb": "CPI", "imf": "PCPIPCH"},  # Primary IMF symbol is PCPIPCH (fallback to PCPI_CH if needed)
        "Growth": {"econdb": "GDP", "imf": "NGDP_RPCH"},      # Primary IMF symbol is NGDP_RPCH (fallback to NGDP_CH if needed)
        "Unemployment": {"econdb": "URATE", "imf": "LUR"}       # Example: unemployment rate
    }
    
    for indicator, symbols in indicators.items():
        print(f"\nFetching {indicator} data:")
        # Try ECONDB first
        df = fetch_data("econdb", symbols["econdb"], country_econdb, start_date, end_date)
        if df.empty:
            print(f"No data from ECONDB for {indicator}, trying IMF...")
            df = fetch_data("imf", symbols["imf"], country_imf, start_date, end_date)
        if not df.empty:
            print(f"\n{indicator} Data:")
            print(df.head())
        else:
            print(f"\nNo {indicator} data found from either provider.")

if __name__ == "__main__":
    main()
