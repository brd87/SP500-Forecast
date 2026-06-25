import yfinance as yf
from datetime import datetime
from pathlib import Path

class RawData:
    # TICKERS = {
    #     "GSPC_sp500": "^GSPC",
    #     "XLK_technology": "XLK",
    #     "XLF_financial": "XLF",
    #     # "XLC_communication": "XLC", # 2018-6-20 (for later experiment)
    #     "XLY_consumer_cyclical": "XLY",
    #     "XLI_industrial": "XLI",
    #     "XLV_healthcare": "XLV",
    #     "XLE_energy": "XLE",
    #     "XLP_consumer_defensive": "XLP",
    #     "XLB_basic_materials": "XLB",
    #     # "XLRE_real_estate": "XLRE", # 2015-10-8 (for later experiment)
    #     "XLU_utilities": "XLU",
    # }

    def __init__(self, start=datetime(1998, 12, 22), fold_path:Path=None):
        tickers = ["^GSPC", "XLK", "XLF", "XLY", "XLI", "XLV", "XLE", "XLP", "XLB", "XLU"]
        
        if start >= datetime(2015, 10, 8): tickers.append("XLRE")
        if start >= datetime(2018, 6, 20): tickers.append("XLC")

        self.data = yf.download(
            tickers,
            start=start,
            group_by="ticker"
        )

        str_start = start.strftime("%Y-%m-%d")
        str_end = datetime.now().strftime("%Y-%m-%d")
        id = datetime.now().strftime("%Y%m%d%H%M%S")
        csv_file_name = f"tick{len(tickers)}_{str_start}_{str_end}_id{id}.csv"
        
        if fold_path is None:
            fold_path = Path(__file__).resolve().parents[2] / "data/raw"
        fold_path.mkdir(exist_ok=True)

        path = fold_path / csv_file_name
        self.data.to_csv(path, index=True)
        self.save_path = path 

        # print(path)
        # print(os.path.dirname(os.path.dirname(__file__)))
        # print(type(self.data))
        # print(self.data.columns)

# data=RawData(start=datetime(2018, 6, 20))
# data=RawData()   
# print(data.data)
