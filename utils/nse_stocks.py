import pandas as pd
import streamlit as st

class NSEStockLookup:
    """Class to handle NSE stock symbol lookup and search functionality"""
    
    def __init__(self):
        self.nse_stocks = self._load_nse_stocks()
    
    def _load_nse_stocks(self):
        """Load NSE stock data with symbols and company names"""
        # Comprehensive list of popular NSE stocks with their Yahoo Finance symbols
        stocks_data = [
            # Banking & Financial Services
            {"symbol": "HDFCBANK.NS", "company": "HDFC Bank Limited", "sector": "Banking"},
            {"symbol": "ICICIBANK.NS", "company": "ICICI Bank Limited", "sector": "Banking"},
            {"symbol": "KOTAKBANK.NS", "company": "Kotak Mahindra Bank Limited", "sector": "Banking"},
            {"symbol": "AXISBANK.NS", "company": "Axis Bank Limited", "sector": "Banking"},
            {"symbol": "SBIN.NS", "company": "State Bank of India", "sector": "Banking"},
            {"symbol": "INDUSINDBK.NS", "company": "IndusInd Bank Limited", "sector": "Banking"},
            {"symbol": "BANKBARODA.NS", "company": "Bank of Baroda", "sector": "Banking"},
            {"symbol": "PNB.NS", "company": "Punjab National Bank", "sector": "Banking"},
            {"symbol": "IDFCFIRSTB.NS", "company": "IDFC First Bank Limited", "sector": "Banking"},
            {"symbol": "FEDERALBNK.NS", "company": "Federal Bank Limited", "sector": "Banking"},
            
            # IT & Technology
            {"symbol": "TCS.NS", "company": "Tata Consultancy Services Limited", "sector": "IT Services"},
            {"symbol": "INFY.NS", "company": "Infosys Limited", "sector": "IT Services"},
            {"symbol": "WIPRO.NS", "company": "Wipro Limited", "sector": "IT Services"},
            {"symbol": "HCLTECH.NS", "company": "HCL Technologies Limited", "sector": "IT Services"},
            {"symbol": "TECHM.NS", "company": "Tech Mahindra Limited", "sector": "IT Services"},
            {"symbol": "LTI.NS", "company": "Larsen & Toubro Infotech Limited", "sector": "IT Services"},
            {"symbol": "MINDTREE.NS", "company": "Mindtree Limited", "sector": "IT Services"},
            {"symbol": "MPHASIS.NS", "company": "Mphasis Limited", "sector": "IT Services"},
            
            # Automobiles
            {"symbol": "MARUTI.NS", "company": "Maruti Suzuki India Limited", "sector": "Automobiles"},
            {"symbol": "TATAMOTORS.NS", "company": "Tata Motors Limited", "sector": "Automobiles"},
            {"symbol": "M&M.NS", "company": "Mahindra & Mahindra Limited", "sector": "Automobiles"},
            {"symbol": "BAJAJ-AUTO.NS", "company": "Bajaj Auto Limited", "sector": "Automobiles"},
            {"symbol": "HEROMOTOCO.NS", "company": "Hero MotoCorp Limited", "sector": "Automobiles"},
            {"symbol": "EICHERMOT.NS", "company": "Eicher Motors Limited", "sector": "Automobiles"},
            {"symbol": "ASHOKLEY.NS", "company": "Ashok Leyland Limited", "sector": "Automobiles"},
            {"symbol": "TVSMOTOR.NS", "company": "TVS Motor Company Limited", "sector": "Automobiles"},
            
            # Pharmaceuticals
            {"symbol": "SUNPHARMA.NS", "company": "Sun Pharmaceutical Industries Limited", "sector": "Pharmaceuticals"},
            {"symbol": "DRREDDY.NS", "company": "Dr. Reddy's Laboratories Limited", "sector": "Pharmaceuticals"},
            {"symbol": "CIPLA.NS", "company": "Cipla Limited", "sector": "Pharmaceuticals"},
            {"symbol": "DIVISLAB.NS", "company": "Divi's Laboratories Limited", "sector": "Pharmaceuticals"},
            {"symbol": "BIOCON.NS", "company": "Biocon Limited", "sector": "Pharmaceuticals"},
            {"symbol": "LUPIN.NS", "company": "Lupin Limited", "sector": "Pharmaceuticals"},
            {"symbol": "AUROPHARMA.NS", "company": "Aurobindo Pharma Limited", "sector": "Pharmaceuticals"},
            {"symbol": "TORNTPHARM.NS", "company": "Torrent Pharmaceuticals Limited", "sector": "Pharmaceuticals"},
            
            # FMCG
            {"symbol": "HINDUNILVR.NS", "company": "Hindustan Unilever Limited", "sector": "FMCG"},
            {"symbol": "ITC.NS", "company": "ITC Limited", "sector": "FMCG"},
            {"symbol": "NESTLEIND.NS", "company": "Nestle India Limited", "sector": "FMCG"},
            {"symbol": "BRITANNIA.NS", "company": "Britannia Industries Limited", "sector": "FMCG"},
            {"symbol": "DABUR.NS", "company": "Dabur India Limited", "sector": "FMCG"},
            {"symbol": "GODREJCP.NS", "company": "Godrej Consumer Products Limited", "sector": "FMCG"},
            {"symbol": "MARICO.NS", "company": "Marico Limited", "sector": "FMCG"},
            
            # Oil & Gas
            {"symbol": "RELIANCE.NS", "company": "Reliance Industries Limited", "sector": "Oil & Gas"},
            {"symbol": "ONGC.NS", "company": "Oil and Natural Gas Corporation Limited", "sector": "Oil & Gas"},
            {"symbol": "IOC.NS", "company": "Indian Oil Corporation Limited", "sector": "Oil & Gas"},
            {"symbol": "BPCL.NS", "company": "Bharat Petroleum Corporation Limited", "sector": "Oil & Gas"},
            {"symbol": "HPCL.NS", "company": "Hindustan Petroleum Corporation Limited", "sector": "Oil & Gas"},
            {"symbol": "GAIL.NS", "company": "GAIL (India) Limited", "sector": "Oil & Gas"},
            
            # Metals & Mining
            {"symbol": "TATASTEEL.NS", "company": "Tata Steel Limited", "sector": "Metals & Mining"},
            {"symbol": "JSWSTEEL.NS", "company": "JSW Steel Limited", "sector": "Metals & Mining"},
            {"symbol": "HINDALCO.NS", "company": "Hindalco Industries Limited", "sector": "Metals & Mining"},
            {"symbol": "VEDL.NS", "company": "Vedanta Limited", "sector": "Metals & Mining"},
            {"symbol": "COALINDIA.NS", "company": "Coal India Limited", "sector": "Metals & Mining"},
            {"symbol": "NMDC.NS", "company": "NMDC Limited", "sector": "Metals & Mining"},
            
            # Telecom
            {"symbol": "BHARTIARTL.NS", "company": "Bharti Airtel Limited", "sector": "Telecom"},
            {"symbol": "IDEA.NS", "company": "Vodafone Idea Limited", "sector": "Telecom"},
            
            # Power
            {"symbol": "POWERGRID.NS", "company": "Power Grid Corporation of India Limited", "sector": "Power"},
            {"symbol": "NTPC.NS", "company": "NTPC Limited", "sector": "Power"},
            {"symbol": "TATAPOWER.NS", "company": "Tata Power Company Limited", "sector": "Power"},
            {"symbol": "ADANIPOWER.NS", "company": "Adani Power Limited", "sector": "Power"},
            
            # Infrastructure
            {"symbol": "LT.NS", "company": "Larsen & Toubro Limited", "sector": "Infrastructure"},
            {"symbol": "UBL.NS", "company": "UltraTech Cement Limited", "sector": "Infrastructure"},
            {"symbol": "GRASIM.NS", "company": "Grasim Industries Limited", "sector": "Infrastructure"},
            {"symbol": "ACC.NS", "company": "ACC Limited", "sector": "Infrastructure"},
            {"symbol": "AMBUJACEMENT.NS", "company": "Ambuja Cements Limited", "sector": "Infrastructure"},
            
            # Retail
            {"symbol": "DMART.NS", "company": "Avenue Supermarts Limited", "sector": "Retail"},
            {"symbol": "TRENT.NS", "company": "Trent Limited", "sector": "Retail"},
            
            # Media & Entertainment
            {"symbol": "ZEEL.NS", "company": "Zee Entertainment Enterprises Limited", "sector": "Media"},
            {"symbol": "SUNTV.NS", "company": "Sun TV Network Limited", "sector": "Media"},
            
            # Real Estate
            {"symbol": "DLF.NS", "company": "DLF Limited", "sector": "Real Estate"},
            {"symbol": "GODREJPROP.NS", "company": "Godrej Properties Limited", "sector": "Real Estate"},
            {"symbol": "OBEROIRLTY.NS", "company": "Oberoi Realty Limited", "sector": "Real Estate"},
            
            # Chemicals
            {"symbol": "UPL.NS", "company": "UPL Limited", "sector": "Chemicals"},
            {"symbol": "PIDILITIND.NS", "company": "Pidilite Industries Limited", "sector": "Chemicals"},
            {"symbol": "ASIANPAINT.NS", "company": "Asian Paints Limited", "sector": "Chemicals"},
            {"symbol": "BERGER.NS", "company": "Berger Paints India Limited", "sector": "Chemicals"},
            
            # Airlines
            {"symbol": "INDIGO.NS", "company": "InterGlobe Aviation Limited", "sector": "Airlines"},
            {"symbol": "SPICEJET.NS", "company": "SpiceJet Limited", "sector": "Airlines"},
            
            # Insurance
            {"symbol": "SBILIFE.NS", "company": "SBI Life Insurance Company Limited", "sector": "Insurance"},
            {"symbol": "HDFCLIFE.NS", "company": "HDFC Life Insurance Company Limited", "sector": "Insurance"},
            {"symbol": "ICICIPRULI.NS", "company": "ICICI Prudential Life Insurance Company Limited", "sector": "Insurance"},
            
            # Adani Group
            {"symbol": "ADANIPORTS.NS", "company": "Adani Ports and Special Economic Zone Limited", "sector": "Infrastructure"},
            {"symbol": "ADANIGREEN.NS", "company": "Adani Green Energy Limited", "sector": "Power"},
            {"symbol": "ADANITRANS.NS", "company": "Adani Transmission Limited", "sector": "Power"},
            {"symbol": "ADANIENSOL.NS", "company": "Adani Energy Solutions Limited", "sector": "Power"},
            
            # Others
            {"symbol": "BAJFINANCE.NS", "company": "Bajaj Finance Limited", "sector": "Financial Services"},
            {"symbol": "BAJAJFINSV.NS", "company": "Bajaj Finserv Limited", "sector": "Financial Services"},
            {"symbol": "SHREECEM.NS", "company": "Shree Cement Limited", "sector": "Cement"},
            {"symbol": "TITAN.NS", "company": "Titan Company Limited", "sector": "Consumer Goods"},
            {"symbol": "MCDOWELL-N.NS", "company": "United Spirits Limited", "sector": "Consumer Goods"},
            {"symbol": "DIXON.NS", "company": "Dixon Technologies (India) Limited", "sector": "Electronics"},
        ]
        
        return pd.DataFrame(stocks_data)
    
    def get_stocks_by_sector(self, sector=None):
        """Get stocks filtered by sector"""
        if sector and sector != "All Sectors":
            return self.nse_stocks[self.nse_stocks['sector'] == sector]
        return self.nse_stocks
    
    def search_stocks(self, query):
        """Search stocks by company name or symbol"""
        if not query:
            return self.nse_stocks
        
        query = query.upper()
        mask = (
            self.nse_stocks['company'].str.upper().str.contains(query, na=False) |
            self.nse_stocks['symbol'].str.upper().str.contains(query, na=False)
        )
        return self.nse_stocks[mask]
    
    def get_all_sectors(self):
        """Get list of all available sectors"""
        return ["All Sectors"] + sorted(self.nse_stocks['sector'].unique().tolist())
    
    def get_symbol_info(self, symbol):
        """Get company info for a given symbol"""
        stock_info = self.nse_stocks[self.nse_stocks['symbol'] == symbol]
        if not stock_info.empty:
            return stock_info.iloc[0].to_dict()
        return None
    
    def get_popular_stocks(self, limit=10):
        """Get most popular/liquid NSE stocks"""
        popular_symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
            "ITC.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "MARUTI.NS"
        ]
        return self.nse_stocks[self.nse_stocks['symbol'].isin(popular_symbols[:limit])]