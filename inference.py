import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the dataset and preprocess
# column_names = ['date', 'receipt_count']
# df = pd.read_csv('data_daily.csv', names=column_names)
# df['date'] = pd.to_datetime(df['date'])
# df.sort_values('date', inplace=True)
#
# data = df[['date', 'receipt_count']].set_index('date')

# Normalized the data in range 0-1
# min_value = data['receipt_count'].min()
# max_value = data['receipt_count'].max()
min_value = 7095414
max_value = 10738865
m = load_model('receipt_count.h5')
# print(f"min_value: {min_value}")
# print(f"max_value: {max_value}")


def inference(month=1):
    start = f'2022-{month}-01'
    end = pd.to_datetime(start) + pd.DateOffset(months=1, days=-1)
    y_2022_dates = pd.date_range(start=start, end=end, freq='D')
    total = 0

    for date in y_2022_dates:
        # print(date)
        future_date = date  # Replace with your specific future date
        future_dates = pd.date_range(end=future_date, periods=10, freq='D')
        future_day_of_week = future_dates.dayofweek.values
        future_month = future_dates.month.values
        future_features = np.column_stack((future_day_of_week, future_month))
        future_features = future_features.reshape((1, 10, 2))
        future_prediction = m.predict(future_features)
        future_prediction_actual = future_prediction * (max_value - min_value) + min_value
        total += future_prediction_actual[0][0]
        print(f'Predicted number of tickets on {future_date}: {future_prediction_actual[0][0]}')
    return total
    # print(f"total receipts in December:{int(total)}")

if __name__ == "__main__":
    print(inference(12))