import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime
import csv

def main():

    X_cols = ['date','Energy (Wh)','lightFixtures','KitchenTemp','KitchenHumidity','LivingRoomTemp','LivingRoomHumidity','LaundryTemp','LaundryHumidity','OfficeRoomTemp','OfficeRoomHumidity','BathRoomTemp','BathRoomHumidity','NorthSideTemp','NorthSideHumidity','IroningRoomTemp','BedRoom1Humidity','BedRoom1RoomTemp','BedRoom1Humidity.1','BedRoom2Temp','BedRoom2Humidity','OutsideTemp','Pressure','OutsideHumdity','Windspeed','Tdewpoint','RandomVariable1','RandomVariable2']
    Y_col = 'Visibility'

    # Training the multivariate regression
    training_data_filename = "./dataset/train.csv"
    training_data = pd.read_csv(training_data_filename)

    # Convert date strings to timestamps
    dates = pd.array(training_data['date'], dtype="string")
    timestamps = []
    for date in dates:
        timestamps.append(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp())
    training_data['date'] = timestamps


    X = training_data[X_cols].iloc[1:11840]
    y = training_data[Y_col].iloc[1:11840]


    regression = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    

    # Testing
    test_data_filename = "./dataset/test.csv"
    test_data = pd.read_csv(test_data_filename)
    test_dates = pd.array(test_data['date'], dtype="string")
    test_timestamps = []
    for date in test_dates:
        test_timestamps.append(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp())
    test_data['date'] = test_timestamps


    predictions = regression.predict(test_data)

    with open('./submission.csv', 'w') as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(predictions)

    
    

    # (this is where we put the predicions in the desired format and save into submission.csv for uploading)

if __name__ == "__main__":
    main()