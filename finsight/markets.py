"""Per-market settings. Select with MARKET=US (default) or MARKET=IN.

Everything that differs between markets lives here: the watchlist, benchmark,
currency, valuation assumptions, trading-day schedule and the RAG corpus
(SEC 10-K filings for the US, company annual-report PDFs for India).
"""

MARKETS = {
    "US": {
        "code": "US",
        "name": "United States",
        "flag": "US",
        "currency": "USD",
        "symbol": "$",
        "big_unit": ("bn", 1e9),
        "benchmark": "SPY",
        "benchmark_name": "SPY",
        "watchlist": ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ"],
        "ticker_suffix": "",
        "starting_cash": 100_000,
        # Valuation: 10y Treasury, US equity risk premium, long-run nominal growth.
        "risk_free": 0.043,
        "equity_risk_premium": 0.055,
        "terminal_growth": 0.025,
        "cost_of_equity_band": (0.07, 0.13),
        # Scheduler: run the crew after the close, local exchange time.
        "timezone": "America/New_York",
        "crew_time": (16, 30),
        "market_hours": ((9, 30), (16, 0)),   # NYSE regular session, New York time
        # RAG corpus.
        "corpus": "sec",
        "doc_label": "10-K",
        "doc_name": "SEC 10-K annual report",
        "collection": "sec_filings",
        "db_suffix": "",
        "rag_alpha": 0.75,  # semantic weight in hybrid re-rank, tuned on eval/rag_eval_set.json
    },
    "IN": {
        "code": "IN",
        "name": "India",
        "flag": "IN",
        "currency": "INR",
        "symbol": "₹",
        "big_unit": ("crore", 1e7),  # Indian convention: 1 crore = 10 million
        "benchmark": "^NSEI",
        "benchmark_name": "NIFTY 50",
        "watchlist": ["WIPRO.NS", "ITC.NS", "SUNPHARMA.NS", "EICHERMOT.NS", "BHARTIARTL.NS", "ASIANPAINT.NS"],
        "ticker_suffix": ".NS",
        "starting_cash": 10_000_000,  # ₹1 crore
        # Valuation: 10y G-sec, Indian equity risk premium, long-run nominal growth.
        "risk_free": 0.065,
        "equity_risk_premium": 0.07,
        "terminal_growth": 0.05,
        "cost_of_equity_band": (0.10, 0.17),
        "timezone": "Asia/Kolkata",
        "crew_time": (16, 0),  # NSE closes 15:30 IST
        "market_hours": ((9, 15), (15, 30)),  # NSE regular session, IST
        "corpus": "pdf",
        "doc_label": "AR",
        "doc_name": "company annual report",
        "collection": "india_annual_reports",
        "db_suffix": "_in",
        "rag_alpha": 0.9,  # tuned on eval/rag_eval_set_in.json
        # Latest annual reports, downloaded from each company's own website
        # by scripts/fetch_annual_reports.py (fiscal year = April-March).
        "reports": {
            "WIPRO.NS": {
                "fy": "2026", "title": "Wipro Integrated Annual Report 2025-26",
                "url": "https://www.wipro.com/content/dam/nexus/staticsites/annual-report-2025/pdf/Integrated-annual-report-2025-26.pdf",
            },
            "ITC.NS": {
                "fy": "2026", "title": "ITC Report and Accounts 2026",
                "url": "https://itcportal.com/content/dam/itc-corporate/open-pdfs/report-and-accounts/ITC-Report-and-Accounts-2026.pdf",
            },
            "SUNPHARMA.NS": {
                "fy": "2026", "title": "Sun Pharma Annual Report 2025-26",
                "url": "https://sunpharma.com/wp-content/uploads/2026/07/Sun-Pharma_AR-2025-26.pdf",
            },
            "EICHERMOT.NS": {
                "fy": "2026", "title": "Eicher Motors Integrated Annual Report 2025-26",
                "url": "https://eicher.in/content/dam/eicher-motors/investor/financial-and-reports/annual-reports/eml-ir-2026-high-res.pdf",
            },
            "BHARTIARTL.NS": {
                "fy": "2026", "title": "Bharti Airtel Integrated Report and Annual Financial Statements 2025-26",
                "url": "https://assets.airtel.in/static-assets/cms/investor/docs/annual_results_2025_26/Airtel---Annual-Report-2026.pdf",
            },
            "ASIANPAINT.NS": {
                "fy": "2026", "title": "Asian Paints Integrated Annual Report 2025-26",
                "url": "https://static.asianpaints.com/content/dam/annual-report-2526/pdf/Asian-Paints-rev-ar-25-26.pdf",
            },
        },
    },
}
