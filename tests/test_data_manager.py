import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
import os
from concurrent.futures import ThreadPoolExecutor
from plato_lite.data_management.data_manager import DataManager


class TestDataManager(unittest.TestCase):

    @patch('plato_lite.data_management.data_manager.create_engine')
    def setUp(self, mock_create_engine):
        self.database_name = 'test_db.db'
        self.dm = DataManager(self.database_name)
        self.mock_create_engine = mock_create_engine


    @patch('pandas.read_csv')
    def test_load_data_csv(self, mock_read_csv):
        file_path = 'data.csv'
        self.dm.load_data(file_path)
        mock_read_csv.assert_called_once_with(file_path)

    @patch('pandas.read_excel')
    def test_load_data_excel(self, mock_read_excel):
        file_path = 'data.xlsx'
        self.dm.load_data(file_path)
        mock_read_excel.assert_called_once_with(file_path, sheet_name=None)

    @patch.object(DataManager, 'load_data')
    def test_load_multiple_files(self, mock_load_data):
        file_paths = ['data1.csv', 'data2.xlsx']
        self.dm.load_multiple_files(file_paths)
        self.assertEqual(mock_load_data.call_count, len(file_paths))

    @patch('pandas.read_sql_table')
    def test_load_table_to_dataframe(self, mock_read_sql_table):
        table_name = 'table1'
        self.dm.load_table_to_dataframe(table_name)
        mock_read_sql_table.assert_called_once_with(table_name, con=self.dm.engine)

    @patch.object(pd.DataFrame, 'to_csv')
    def test_save_data_csv(self, mock_to_csv):
        df = pd.DataFrame()
        file_path = 'data.csv'
        self.dm.save_data(df, file_path)
        mock_to_csv.assert_called_once_with(file_path)

    @patch.object(pd.DataFrame, 'to_excel')
    def test_save_data_excel(self, mock_to_excel):
        df = pd.DataFrame()
        file_path = 'data.xlsx'
        self.dm.save_data(df, file_path)
        mock_to_excel.assert_called_once_with(file_path)


if __name__ == '__main__':
    unittest.main()