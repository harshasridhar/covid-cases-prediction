import pandas as pd
from requests import get
from time import time
from constants import *
import logging
from os.path import exists

logger = logging.getLogger(__name__)


class DataUtils:
    state_data: pd.DataFrame = None
    country_data: pd.DataFrame = None
    data = None

    @staticmethod
    def get_state_data():
        if DataUtils.state_data is None:
            DataUtils.state_data = DataUtils.read_data(preprocess_data=True)
        return DataUtils.state_data

    @staticmethod
    def get_country_data() -> pd.DataFrame:
        if DataUtils.country_data is None:
            DataUtils.country_data = DataUtils.read_data(level='country')
        return DataUtils.country_data

    @staticmethod
    def download_data(save_to_data_dir=False, level='state'):
        print('Downloading Data')
        start_time = time()
        response = get(DATA_DOWNLOAD_URL[level])
        end_time = time()
        logger.info("Downloaded file", extra={'time_taken': end_time - start_time,
                                              'status_code': response.status_code,
                                              'level': level})
        if 'state' in level:
            response_data = response.json().get('rows', [])
            collated_data = collated_data = pd.DataFrame(
                columns=['Date', 'state', 'cured', 'death', 'confirmed', 'confirmed_india', 'confirmed_foreign'])

            state_code_to_name_map = {value: key for key, value in states.items()}
            print('Processing')
            start_time = time()
            for index, row in enumerate(response_data):
                row = row['value']
                collated_data.loc[index] = [row['report_time'],
                                            state_code_to_name_map.get(row['state'].upper(), row['state']),
                                            row['cured'], row['death'], row['confirmed'],
                                            row.get('confirmed_india', None), row.get('confirmed_foreign', None)]
            end_time = time()
            logger.info("Processed file", extra={'time_taken': end_time - start_time, 'num_rows': len(response_data)})
            if save_to_data_dir:
                collated_data.to_csv(DATA_LOCATION[level], index=False)
            return collated_data
        elif 'country' in level:
            response_data = response.json()
            f = open('all_totals.csv', 'w')
            f.write('Date,key,value\n')
            for row in response_data.get('rows', []):
                f.write('{},{},{}\n'.format(row['key'][0], row['key'][1], row['value']))
            f.close()
            data = pd.read_csv('all_totals.csv')
            converted = pd.DataFrame(columns=['Date', 'active_cases', 'cured', 'death', 'total_confirmed_cases'])
            converted['Date'] = data['Date'].unique()
            converted.set_index('Date', inplace=True)
            for dt in converted.index:
                values = []
                for col in converted.columns:
                    values.append(data[(data['Date'] == dt) & (data['key'] == col)]['value'].values[0])
                converted.loc[dt] = values
            if save_to_data_dir:
                converted.to_csv(DATA_LOCATION[level])
            return converted

    @staticmethod
    def read_data(preprocess_data=False, level='state'):
        logger.info("Reading Data")
        level = level.lower().strip()
        path = DATA_LOCATION[level]
        if not exists(path):
            DataUtils.download_data(save_to_data_dir=True, level=level)
        data = pd.read_csv(path, parse_dates=['Date'])
        data['Date'] = pd.to_datetime(data['Date'].dt.date)
        if 'state' in level and preprocess_data:
            data = data \
                .groupby('state').apply(DataUtils.__handle_group).reset_index(). \
                drop(columns='level_1')
        return data

    @staticmethod
    def __handle_group(group_data):
        group_data = group_data.groupby('Date').last()
        group_data = group_data.reindex(pd.date_range(start=group_data.index[0], end=group_data.index[-1], freq='1d'),
                                        method='ffill')
        group_data.reset_index(inplace=True)
        group_data.rename(columns={'index': 'Date'}, inplace=True)
        return pd.concat([group_data[['Date']], group_data.drop(columns=['Date', 'state']).diff()], axis=1)
