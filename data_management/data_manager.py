import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import os
from utils.logger import PlatoLogger

logger = PlatoLogger().logger


class DataManager:
    def __init__(self, db_name='plato_lite.db'):
        self.engine = create_engine(f'sqlite:///{db_name}',
                                    connect_args={'check_same_thread': False},
                                    poolclass=StaticPool)
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    def load_data(self, file_path, sheet_name=None, table_name=None, save_to_db=False, **kwargs):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            else:
                raise ValueError("Unsupported file format")

            if save_to_db:
                self._save_to_db(df, table_name or os.path.splitext(os.path.basename(file_path))[0])

            logger.info(f"Data loaded from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def load_multiple_files(self, file_paths, sheet_names=None, table_names=None, save_to_db=False, **kwargs):
        futures = [self.executor.submit(self.load_data, fp,
                                        sheet_names[i] if sheet_names else None,
                                        table_names[i] if table_names else None,
                                        save_to_db, **kwargs)
                   for i, fp in enumerate(file_paths)]
        return [future.result() for future in futures]

    def _save_to_db(self, df, table_name):
        chunk_size = 2000
        for i in range(0, len(df), chunk_size):
            df.iloc[i:i + chunk_size].to_sql(table_name, con=self.engine, if_exists='append', index=False,
                                             method='multi')
        logger.info(f"Data saved to table {table_name}")

    def execute_query(self, query):
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                logger.info(f"Query executed: {query}")
                return result.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {query}: {e}")

    def load_table_to_dataframe(self, table_name):
        try:
            df = pd.read_sql_table(table_name, con=self.engine)
            logger.info(f"Table {table_name} loaded into DataFrame")
            return df
        except Exception as e:
            logger.error(f"Error loading table {table_name} into DataFrame: {e}")
            return None

    def save_data(self, df, file_path, **kwargs):
        try:
            if file_path.endswith('.csv'):
                df.to_csv(file_path, **kwargs)
            elif file_path.endswith(('.xls', '.xlsx')):
                df.to_excel(file_path, **kwargs)
            else:
                raise ValueError("Unsupported file format")

            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise


class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self, drop_duplicates=True, fill_strategy='mean', drop_na=False,
                   remove_outliers=False, outlier_method='IQR', outlier_factor=1.5):
        if drop_duplicates:
            self.df.drop_duplicates(inplace=True)

        if fill_strategy:
            self._fill_missing_values(fill_strategy)

        if drop_na:
            self.df.dropna(inplace=True)

        if remove_outliers:
            self._remove_outliers(method=outlier_method, factor=outlier_factor)

        logger.info("Data cleaning completed")
        return self

    def _fill_missing_values(self, strategy):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if strategy in ['mean', 'median']:
            self.df[numeric_cols] = self.df[numeric_cols].apply(
                lambda x: x.fillna(getattr(x, strategy)()))
        elif strategy == 'mode':
            self.df = self.df.apply(lambda x: x.fillna(x.mode()[0]))
        else:
            self.df.fillna(strategy, inplace=True)

    def _remove_outliers(self, method='IQR', factor=1.5):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if method == 'IQR':
            Q1 = self.df[numeric_cols].quantile(0.25)
            Q3 = self.df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[~((self.df[numeric_cols] < (Q1 - factor * IQR)) |
                                (self.df[numeric_cols] > (Q3 + factor * IQR))).any(axis=1)]
        elif method == 'Z-score':
            z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
            self.df = self.df[(z_scores < factor).all(axis=1)]

    def transform_data(self, encode_columns=None, scale_columns=None, log_columns=None,
                       bin_columns=None, bins=None, labels=None):
        if encode_columns:
            self._encode_labels(encode_columns)

        if scale_columns:
            self._scale_data(scale_columns)

        if log_columns:
            self.df[log_columns] = np.log1p(self.df[log_columns])

        if bin_columns and bins:
            self.df[bin_columns] = pd.cut(self.df[bin_columns], bins=bins, labels=labels)

        logger.info("Data transformation completed")
        return self

    def _encode_labels(self, columns):
        encoder = OrdinalEncoder(cols=columns)
        self.df = encoder.fit_transform(self.df)

    def _scale_data(self, columns):
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])

    def get_processed_data(self):
        return self.df


class QueryBuilder:
    def __init__(self):
        self.query = ""

    def select(self, columns="*"):
        self.query = f"SELECT {columns} "
        return self

    def from_table(self, table_name):
        self.query += f"FROM {table_name} "
        return self

    def where(self, condition):
        self.query += f"WHERE {condition} "
        return self

    def group_by(self, columns):
        self.query += f"GROUP BY {columns} "
        return self

    def having(self, condition):
        self.query += f"HAVING {condition} "
        return self

    def order_by(self, columns, order="ASC"):
        self.query += f"ORDER BY {columns} {order} "
        return self

    def build(self):
        return self.query.strip()