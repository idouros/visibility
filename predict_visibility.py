import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime

def date_strings_to_timestamps(training_data):
    dates = pd.array(training_data['date'], dtype="string")
    timestamps = []
    for date in dates:
        timestamps.append(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp())  
    return timestamps

def main():

    X_cols = ['date','Energy (Wh)','lightFixtures','KitchenTemp','KitchenHumidity','LivingRoomTemp','LivingRoomHumidity','LaundryTemp','LaundryHumidity','OfficeRoomTemp','OfficeRoomHumidity','BathRoomTemp','BathRoomHumidity','NorthSideTemp','NorthSideHumidity','IroningRoomTemp','BedRoom1Humidity','BedRoom1RoomTemp','BedRoom1Humidity.1','BedRoom2Temp','BedRoom2Humidity','OutsideTemp','Pressure','OutsideHumdity','Windspeed','Tdewpoint','RandomVariable1','RandomVariable2']
    Y_col = 'Visibility'

    # Get the training data
    training_data_filename = "./dataset/train.csv"
    training_data = pd.read_csv(training_data_filename)

    # Prepare the data, part 1: Convert date strings to timestamps
    training_data['date'] = date_strings_to_timestamps(training_data)

    # Prepare the data, part 2: Remove invalid traning data (with predicate missing)
    training_data = training_data.loc[training_data[Y_col].notnull()]

    # Train the regression
    num_rows = training_data.shape[0]
    X = training_data[X_cols].iloc[1:num_rows]
    y = training_data[Y_col].iloc[1:num_rows]
    regression = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)

    # Test
    test_data_filename = "./dataset/test.csv"
    test_data = pd.read_csv(test_data_filename)
    test_dates = test_data['date']
    test_data['date'] = date_strings_to_timestamps(test_data)
    predictions = regression.predict(test_data)

    # Output the results in the specified format (as per sample_submission.csv)
    test_dates_out = pd.DataFrame(test_dates, columns=['date'])
    predicted_visibilities = pd.DataFrame(predictions, columns=['Visibility'])
    predictions_out = pd.concat([test_dates_out, predicted_visibilities], ignore_index=False, axis=1)
    predictions_out.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()