import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime

class DataProcessing:
    def __init__(self, data_df:pd.DataFrame=None, csv_path:Path=None):

        if data_df is None:
            try:
                data_df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=["Date"])
            except Exception:
                raise Exception("ERROR : 'data_path' is of type None or incorrect")

        self.raw_data = data_df
        #print(self.raw_data)
        self.preprocessed, self.dsready = self._calculate_features()
        self.save_preprocessed_path, self.save_dsready_path = self._save_features()

    def _calculate_features(self):

        feature_frames = []
        tickers = self.raw_data.columns.get_level_values(0).unique()
        price_col = "Close"
        if "^GSPC" not in tickers:
            raise ValueError("^GSPC ticker is missing from dataset")
        
        for ticker in tickers:
            ticker_df = self.raw_data[ticker].copy()

            close = ticker_df[price_col]
            
            feat = pd.DataFrame(index=close.index)

            feat[f"{ticker}_ret_1d"] = close.pct_change()
            feat[f"{ticker}_ret_5d"] = close.pct_change(5)
            feat[f"{ticker}_ret_20d"] = close.pct_change(20)
            feat[f"{ticker}_ret_60d"] = close.pct_change(60)

            feat[f"{ticker}_ma20_norm"] = close.rolling(20).mean() / close
            feat[f"{ticker}_ma50_norm"] = close.rolling(50).mean() / close
            feat[f"{ticker}_ma200_norm"] = close.rolling(200).mean() / close

            #rolling annualized volatility
            feat[f"{ticker}_vol20"] = (
                feat[f"{ticker}_ret_1d"]
                .rolling(20)
                .std()
                * np.sqrt(252)
            )

            feat[f"{ticker}_vol60"] = (
                feat[f"{ticker}_ret_1d"]
                .rolling(60)
                .std()
                * np.sqrt(252)
            )

            feature_frames.append(feat)

        preprocessed = pd.concat(feature_frames, axis=1)
        preprocessed.index.name = "Date"

        gspc = self.raw_data["^GSPC"].copy()
        gspc_close = gspc[price_col]
        future_ret_5d = gspc_close.shift(-5) / gspc_close - 1 #might redifine in the future

        #noise reduction
        thr = 0.002
        target = np.where(future_ret_5d > thr, 1,
                np.where(future_ret_5d < -thr, 0, np.nan))

        # preprocessed["gspc_future_ret_5d"] = future_ret_5d
        preprocessed["gspc_target_up_5d"] = target

        dsready = preprocessed.dropna().sort_index()

        return preprocessed, dsready

    def _save_features(self, save_preprocessed_path:Path=None, save_dsready_path:Path=None):
        if save_preprocessed_path is None:
            save_preprocessed_path = Path(__file__).resolve().parents[2] / "data/preprocessed"
        if save_dsready_path is None:
            save_dsready_path = Path(__file__).resolve().parents[2] / "data/dsready"
            
        save_preprocessed_path.mkdir(exist_ok=True)
        save_dsready_path.mkdir(exist_ok=True)

        id = datetime.now().strftime("%Y%m%d%H%M%S")
        len_features = len(self.raw_data.columns.get_level_values(0).unique())

        start_date = self.preprocessed.index.min().strftime("%Y-%m-%d")
        end_date = self.preprocessed.index.max().strftime("%Y-%m-%d")
        file_name = f"features{len_features}_{start_date}_{end_date}_id{id}.csv"
        save_preprocessed_path = save_preprocessed_path / file_name

        start_date = self.dsready.index.min().strftime("%Y-%m-%d")
        end_date = self.dsready.index.max().strftime("%Y-%m-%d")
        file_name = f"features{len_features}_{start_date}_{end_date}_id{id}.csv"
        save_dsready_path = save_dsready_path / file_name

        self.preprocessed.to_csv(save_preprocessed_path)
        self.dsready.to_csv(save_dsready_path)
        return save_preprocessed_path, save_dsready_path

#data = DataProcessing(csv_path="D:/projectsGYM/SP500FC/SP500-Forecast/data/raw/tick10_1998-12-22_2026-06-14.csv")