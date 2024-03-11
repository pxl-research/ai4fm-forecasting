import difflib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

RAW_DATA_PATH = "../data/raw"
CLEAN_DATA_PATH = "../data/clean"

# ----------------------------------------------------------------------------------------------------------------------
# Weather_wsc11
# ----------------------------------------------------------------------------------------------------------------------
def dfs_dict_weather_wsc11_cleanup(selected_features: list[str], lokalen_list: list[str], path: str = RAW_DATA_PATH) -> tuple[dict[str, pd.DataFrame], np.datetime64, np.datetime64]:
    df = pd.read_csv(f"{path}/weather_wsc11.csv", dtype={"id": "Int64", "raam_id": "string", "timestamp": "string", 
                                                    "msg_timestamp": "Int64", "rWindSpeed": "float32", "rWindSpeedMean": "float32",
                                                    "rAirTemperature": "float32", "rHousingTemperature": "float32",
                                                    "rDewPointTemperature": "float32", "rRelativeHumidity": "float32", 
                                                    "rAbsoluteHumitidy": "float32", "rAbsoluteAirPressure": "float32", 
                                                    "rRelativeAirPressure": "float32", "rGlobalRadiation": "float32", 
                                                    "rBrightnessNorth": "float32", "rBrightnessEast": "float32", 
                                                    "rBrightnessSouth": "float32", "rBrightnessWest": "float32", 
                                                    "udiTwilight": "float32", "bPrecipitation": "boolean",
                                                    "udiDate": "float32", "udiTime": "float32", "diTimeFormat": "float32", 
                                                    "rLongitude": "float32", "rLatitude": "float32",
                                                    "rSunElevation": "float32", "rSunAzimuth": "float32", 
                                                    "udiHeightAboveSea": "float32"}
                                                    , parse_dates=["timestamp"])
    
    # Change datatype from boolean to float32 (replace True with 1.0 and False with 0.0)
    df["bPrecipitation"] = df["bPrecipitation"].astype("float32")
    
    # Timestamps (insertion in database) match msg_timestamps (Unix timestamp).
    # Only "timestamp" is guaranteed to be non-missing.
    df["DateTime"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("DateTime", inplace=True)
    df.drop("timestamp", axis=1, inplace=True)
    df.drop("msg_timestamp", axis=1, inplace=True)

    # Only select necessary columns before downsampling and imputation 
    df = df[selected_features]
    # Skip a few rows until we get to first minutes divisible by 10
    first_divisible_index = df.index[df.index.minute % 10 == 0].min()
    # Downsample to values each 10 minutes by averaging except for boolean column, also ignore NA or NaN values in mean computation
    df_prec = df.loc[first_divisible_index:, df.columns == "bPrecipitation"].resample(rule="10T").max()
    df = df.loc[first_divisible_index:, df.columns != "bPrecipitation"].resample(rule="10T").mean()
    df = pd.merge(left=df, right=df_prec, on="DateTime", how="inner")
    # Perform MICE imputation (Multiple Imputation by Chained Equations)
    # models each feature with missing values as a function of other features, and uses that estimate for imputation. 
    # It does so in an iterated round-robin fashion: at each step, a feature column is designated as output y and the 
    # other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y. 
    # Then, the regressor is used to predict the missing values of y. This is done for each feature in an iterative fashion, 
    # and then is repeated for max_iter imputation rounds. 
    # The results of the final imputation round are returned.
    imputer = IterativeImputer(max_iter=10, initial_strategy="mean", imputation_order="ascending")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    # Used to create common window
    start_datetime = df.index.min()
    end_datetime = df.index.max()
    # Makes merging process of individual tables to 1 table easier
    dfs_dict = {lokaal: df for lokaal in lokalen_list}

    # Rename all columns so they become unique per table before merging (to avoid duplicate column names)
    tablename = "weather_wsc11"
    for key, df in dfs_dict.items():
        df.columns = [f"{tablename}_{col}" for col in df.columns]
        dfs_dict[key] = df

    return dfs_dict, start_datetime, end_datetime


# ----------------------------------------------------------------------------------------------------------------------
# Radiatoren
# ----------------------------------------------------------------------------------------------------------------------
def dfs_dict_radiatoren_cleanup(selected_features: list[str], lokalen_list: list[str], path: str = RAW_DATA_PATH) -> tuple[dict[str, pd.DataFrame], np.datetime64, np.datetime64]:
    df_lokalen_radiatoren = pd.read_csv(f"{path}/lokalen_radiatoren.csv", dtype={"lokaal_naam": "string", "radiator_id": "Int64"})
    #df_lokalen_radiatoren = df_lokalen_radiatoren[df_lokalen_radiatoren["lokaal_naam"].isin(lokalen_list)]
    df_lokalen_radiatoren = df_lokalen_radiatoren[df_lokalen_radiatoren["lokaal_naam"].str.contains('|'.join(lokalen_list), case=False)]
    to_replace_list = sorted(df_lokalen_radiatoren["lokaal_naam"].unique().tolist())
    replacement_list = sorted(lokalen_list)
    mapping_dict = dict(zip(to_replace_list, replacement_list))
    df_lokalen_radiatoren["lokaal_naam"] = df_lokalen_radiatoren["lokaal_naam"].replace(mapping_dict, regex=True)

    df = pd.read_csv(f"{path}/radiatoren.csv", dtype={"id": "Int64", "radiator_id": "Int64", "tijdstip": "string", 
                                                    "msg_timestamp": "Int64", "debiet": "float32", "aanvoer_temp": "float32", 
                                                    "retour_temp": "float32", "delta_t": "float32", "vermogen": "float32", 
                                                    "positie_kraan": "float32", "energieteller": "float32"}
                                                    , parse_dates=["tijdstip"])

    # Timestamps (insertion in database) match msg_timestamps (Unix timestamp).
    # Only "tijdstip" is guaranteed to be non-missing.
    df["DateTime"] = pd.to_datetime(df["tijdstip"], unit="s")
    df.set_index("DateTime", inplace=True)
    df.drop("tijdstip", axis=1, inplace=True)
    df.drop("msg_timestamp", axis=1, inplace=True)
    
    # Join so sensors get mapped to "lokalen" (classrooms)
    df["DateTime"] = df.index
    df = pd.merge(left=df, right=df_lokalen_radiatoren, on="radiator_id", how="inner")
    df.set_index("DateTime", inplace=True)
    # Group by and split into a dictionary
    dfs_dict = {key: group for key, group in df.groupby("lokaal_naam")}
    
    # Find the common DateTime indices window
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values())
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values())
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    for key, df in dfs_dict.items():
        # Aggregate multiple sensors on the same DateTime
        grouped = df.groupby(df.index)
        df = grouped.mean(numeric_only=True)
        # Crop DataFrame to be in common indices range
        df = df.loc[(df.index >= start_datetime) & (df.index <= end_datetime)]
        # Only select necessary columns before downsampling and imputation 
        df = df[selected_features]
        # Skip a few rows until we get to first minutes divisible by 10
        first_divisible_index = df.index[df.index.minute % 10 == 0].min()
        # Downsample to values each 10 minutes by averaging, also ignore NA or NaN values in mean computation
        df = df.loc[first_divisible_index:].resample(rule="10T").mean()
        # NOTE: [DEBUG] Check if consecutive DateTime's have differences equal to 10 minutes
        """
        expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="10T")
        missing_indices = expected_index.difference(df.index)
        missing_indices_count = missing_indices.shape[0]
        if missing_indices_count > 0:
            raise AssertionError()    
        time_diff = df.index.to_series().diff()
        is_correct_interval = all(time_diff[1:] == pd.Timedelta(minutes=10))
        if not is_correct_interval:
            raise AssertionError()
        """
        # MICE Imputation
        imputer = IterativeImputer(max_iter=10, initial_strategy="mean", imputation_order="ascending")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        # Mutate the DataFrame in the dictionary
        dfs_dict[key] = df

    # Rename all columns so they become unique per table before merging (to avoid duplicate column names)
    tablename = "radiatoren"
    for key, df in dfs_dict.items():
        df.columns = [f"{tablename}_{col}" for col in df.columns]
        dfs_dict[key] = df

    return dfs_dict, start_datetime, end_datetime


# ----------------------------------------------------------------------------------------------------------------------
# Ventilatie
# ----------------------------------------------------------------------------------------------------------------------
def dfs_dict_ventilatie_cleanup(selected_features: list[str], lokalen_list: list[str], path: str = RAW_DATA_PATH) -> tuple[dict[str, pd.DataFrame], np.datetime64, np.datetime64]:
    df_lokalen_ventilatie = pd.read_csv(f"{path}/lokalen_ventilatie.csv", dtype={"lokaal_naam": "string", "ventilatie_id": "string"})
    df_lokalen_ventilatie = df_lokalen_ventilatie[df_lokalen_ventilatie["lokaal_naam"].str.contains('|'.join(lokalen_list), case=False)]
    to_replace_list = sorted(df_lokalen_ventilatie["lokaal_naam"].unique().tolist())
    replacement_list = sorted(lokalen_list)
    # Make sure only classrooms with a ventilation system are selected
    # Split the boolean list into n pieces and perform OR operation on each piece to obtain a single boolean list.
    bool_list = [item in t for t in to_replace_list for item in replacement_list]
    def split_and_or(boolean_list: list[bool], n: int) -> list[bool]:
        piece_length = len(boolean_list) // n
        boolean_pieces = [boolean_list[i:i + piece_length] for i in range(0, len(boolean_list), piece_length)]
        return [any(piece) for piece in zip(*boolean_pieces)]
    bool_list = split_and_or(bool_list, len(to_replace_list))
    true_indices = [index for index, value in enumerate(bool_list) if value]
    replacement_list = [replacement_list[index] for index in true_indices]
    mapping_dict = dict(zip(to_replace_list, replacement_list))
    
    df_lokalen_ventilatie["lokaal_naam"] = df_lokalen_ventilatie["lokaal_naam"].replace(mapping_dict, regex=True)

    df = pd.read_csv(f"{path}/ventilatie.csv", dtype={"id": "Int64", "ventilatie_id": "string", "tijdstip": "string", 
                                                    "msg_timestamp": "Int64", "aanvoersnelheid": "float32", 
                                                    "afvoersnelheid": "float32", "aanvoer_temp": "float32", "afvoer_temp": "float32", 
                                                    "buiten_temp": "float32", "co2": "float32", "rel_vochtigheid": "float32", 
                                                    "modus": "string", "filtertijd_resterend": "float32"}
                                                    , parse_dates=["tijdstip"])

    # Timestamps (insertion in database) match msg_timestamps (Unix timestamp).
    # Only "tijdstip" is guaranteed to be non-missing.
    df["DateTime"] = pd.to_datetime(df["tijdstip"], unit="s")
    df.set_index("DateTime", inplace=True)
    df.drop("tijdstip", axis=1, inplace=True)
    df.drop("msg_timestamp", axis=1, inplace=True)
    
    # Join so sensors get mapped to "lokalen" (classrooms)
    df["DateTime"] = df.index
    df = pd.merge(left=df, right=df_lokalen_ventilatie, on="ventilatie_id", how="inner")
    df.set_index("DateTime", inplace=True)
    # Group by and split into a dictionary
    dfs_dict = {key: group for key, group in df.groupby("lokaal_naam")}

    # Find the common DateTime indices window
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values())
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values())
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    for key, df in dfs_dict.items():
        # Aggregate multiple sensors on the same DateTime
        grouped = df.groupby(df.index)
        df = grouped.mean(numeric_only=True)
        # Crop DataFrame to be in common indices range
        df = df.loc[(df.index >= start_datetime) & (df.index <= end_datetime)]
        # Only select necessary columns before downsampling and imputation 
        df = df[selected_features]
        # Skip a few rows until we get to first minutes divisible by 10
        first_divisible_index = df.index[df.index.minute % 10 == 0].min()
        # Downsample to values each 10 minutes by averaging, also ignore NA or NaN values in mean computation
        df = df.loc[first_divisible_index:].resample(rule="10T").mean()
        # MICE Imputation
        imputer = IterativeImputer(max_iter=10, initial_strategy="mean", imputation_order="ascending")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        # Mutate the DataFrame in the dictionary
        dfs_dict[key] = df

    # Find the common DateTime indices window: again after resampling everything
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values()).to_datetime64()
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values()).to_datetime64()
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    # Add empty DataFrame for classrooms without ventilation and sort keys by classroom order in lokalen_list   
    keys_without_ventilation = list(set(lokalen_list) - set(dfs_dict.keys()))
    for k in keys_without_ventilation:
        dfs_dict[k] = pd.DataFrame()
    dfs_dict = dict(sorted(dfs_dict.items(), key=lambda x: lokalen_list.index(x[0])))

    # Rename all columns so they become unique per table before merging (to avoid duplicate column names)
    tablename = "ventilatie"
    for key, df in dfs_dict.items():
        df.columns = [f"{tablename}_{col}" for col in df.columns]
        dfs_dict[key] = df
    
    return dfs_dict, start_datetime, end_datetime


# ----------------------------------------------------------------------------------------------------------------------
# Klimaat
# ----------------------------------------------------------------------------------------------------------------------
def dfs_dict_klimaat_cleanup(selected_features: list[str], lokalen_list: list[str], path: str = RAW_DATA_PATH) -> tuple[dict[str, pd.DataFrame], np.datetime64, np.datetime64]:
    df_lokalen_klimaat = pd.read_csv(f"{path}/lokalen_klimaat.csv", dtype={"lokaal_naam": "string", "sensor_id": "string"})

    df_lokalen_klimaat = df_lokalen_klimaat[df_lokalen_klimaat["lokaal_naam"].str.contains('|'.join(lokalen_list), case=False)]
    to_replace_list = sorted(df_lokalen_klimaat["lokaal_naam"].unique().tolist())
    replacement_list = sorted(lokalen_list)
    mapping_dict = dict(zip(to_replace_list, replacement_list))
    df_lokalen_klimaat["lokaal_naam"] = df_lokalen_klimaat["lokaal_naam"].replace(mapping_dict, regex=True)

    df = pd.read_csv(f"{path}/klimaat.csv", dtype={"id": "Int64", "sensor_id": "string", "tijdstip": "string", 
                                                    "msg_timestamp": "Int64", "temperatuur": "float32", "relatieve_vochtigheid": "float32", "co2": "float32"}
                                                    , parse_dates=["tijdstip"])

    # Timestamps (insertion in database) match msg_timestamps (Unix timestamp).
    # Only "tijdstip" is guaranteed to be non-missing.
    df["DateTime"] = pd.to_datetime(df["tijdstip"], unit="s")
    df.set_index("DateTime", inplace=True)
    df.drop("tijdstip", axis=1, inplace=True)
    df.drop("msg_timestamp", axis=1, inplace=True)

    # Join so sensors get mapped to "lokalen" (classrooms)
    df["DateTime"] = df.index
    df = pd.merge(left=df, right=df_lokalen_klimaat, on="sensor_id", how="inner")
    df.set_index("DateTime", inplace=True)
    # Group by and split into a dictionary
    dfs_dict = {key: group for key, group in df.groupby("lokaal_naam")}
    
    # Find the common DateTime indices window
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values())
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values())
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    for key, df in dfs_dict.items():
        # Aggregate multiple sensors on the same DateTime
        grouped = df.groupby(df.index)
        df = grouped.mean(numeric_only=True)
        # Crop DataFrame to be in common indices range
        df = df.loc[(df.index >= start_datetime) & (df.index <= end_datetime)]
        # Only select necessary columns before downsampling and imputation 
        df = df[selected_features]    
        # Skip a few rows until we get to first minutes divisible by 10
        first_divisible_index = df.index[df.index.minute % 10 == 0].min()
        # Downsample to values each 10 minutes by averaging, also ignore NA or NaN values in mean computation
        df = df.loc[first_divisible_index:].resample(rule="10T").mean()        
        # MICE Imputation
        imputer = IterativeImputer(max_iter=10, initial_strategy="mean", imputation_order="ascending")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        # Mutate the DataFrame in the dictionary
        dfs_dict[key] = df

    # Find the common DateTime indices window: again after resampling everything
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values()).to_datetime64()
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values()).to_datetime64()
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    # Rename all columns so they become unique per table before merging (to avoid duplicate column names)
    tablename = "klimaat"
    for key, df in dfs_dict.items():
        df.columns = [f"{tablename}_{col}" for col in df.columns]
        dfs_dict[key] = df

    return dfs_dict, start_datetime, end_datetime


# ----------------------------------------------------------------------------------------------------------------------
# Personentellers
# ----------------------------------------------------------------------------------------------------------------------
def dfs_dict_personentellers_cleanup(selected_features: list[str], lokalen_list: list[str], path: str = RAW_DATA_PATH) -> tuple[dict[str, pd.DataFrame], np.datetime64, np.datetime64]:
    df_lokalen_personentellers = pd.read_csv(f"{path}/lokalen_personentellers.csv", dtype={"lokaal_naam": "string", "personenteller_id": "string"})

    df_lokalen_personentellers = df_lokalen_personentellers[df_lokalen_personentellers["lokaal_naam"].str.contains('|'.join(lokalen_list), case=False)]
    to_replace_list = sorted(df_lokalen_personentellers["lokaal_naam"].unique().tolist())
    replacement_list = sorted(lokalen_list)
    mapping_dict = dict(zip(to_replace_list, replacement_list))
    df_lokalen_personentellers["lokaal_naam"] = df_lokalen_personentellers["lokaal_naam"].replace(mapping_dict, regex=True)
    
    df = pd.read_csv(f"{path}/personentellers.csv", dtype={"id": "Int64", "personenteller_id": "string", "tijdstip": "string", 
                                                    "msg_timestamp": "Int64", "aantal_personen": "float32", "temperatuur": "float32",
                                                    "vochtigheid": "float32", "helderheid": "float32"}
                                                    , parse_dates=["tijdstip"])
    # Timestamps (insertion in database) match msg_timestamps (Unix timestamp).
    # Only "tijdstip" is guaranteed to be non-missing.
    df["DateTime"] = pd.to_datetime(df["tijdstip"], unit="s")
    df.set_index("DateTime", inplace=True)
    df.drop("tijdstip", axis=1, inplace=True)
    df.drop("msg_timestamp", axis=1, inplace=True)
    
    # Join so sensors get mapped to "lokalen" (classrooms)
    df["DateTime"] = df.index
    df = pd.merge(left=df, right=df_lokalen_personentellers, on="personenteller_id", how="inner")
    df.set_index("DateTime", inplace=True)
    # Group by and split into a dictionary
    dfs_dict = {key: group for key, group in df.groupby("lokaal_naam")}
    
    # Find the common DateTime indices window
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values())
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values())
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    for key, df in dfs_dict.items():
        # Aggregate multiple sensors on the same DateTime
        grouped = df.groupby(df.index)
        df = grouped.mean(numeric_only=True)
        # Crop DataFrame to be in common indices range
        df = df.loc[(df.index >= start_datetime) & (df.index <= end_datetime)]
        # Only select necessary columns before downsampling and imputation 
        df = df[selected_features]
        # Skip a few rows until we get to first minutes divisible by 10
        first_divisible_index = df.index[df.index.minute % 10 == 0].min()
        # Downsample to values each 10 minutes by averaging, also ignore NA or NaN values in mean computation
        df = df.loc[first_divisible_index:].resample(rule="10T").mean()
        # Single imputation by mean substitution
        imputer = SimpleImputer(strategy="mean")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        # Mutate the DataFrame in the dictionary
        dfs_dict[key] = df

    # Find the common DateTime indices window: again after resampling everything
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values()).to_datetime64()
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values()).to_datetime64()
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    # Rename all columns so they become unique per table before merging (to avoid duplicate column names)
    tablename = "personentellers"
    for key, df in dfs_dict.items():
        df.columns = [f"{tablename}_{col}" for col in df.columns]
        dfs_dict[key] = df

    return dfs_dict, start_datetime, end_datetime


# ----------------------------------------------------------------------------------------------------------------------
# Ramen
# ----------------------------------------------------------------------------------------------------------------------
def dfs_dict_ramen_cleanup(selected_features: list[str], lokalen_list: list[str], path: str = RAW_DATA_PATH) -> tuple[dict[str, pd.DataFrame], np.datetime64, np.datetime64]:
    df = pd.read_csv(f"{path}/ramen.csv", dtype={"id": "Int64", "raam_id": "string", "tijdstip": "string", 
                                                    "msg_timestamp": "Int64", "raamstand": "string", "raamopen": "Int8"}
                                                    , parse_dates=["tijdstip"])

    # Timestamps (insertion in database) match msg_timestamps (Unix timestamp).
    # Only "tijdstip" is guaranteed to be non-missing.
    df["DateTime"] = pd.to_datetime(df["tijdstip"], unit="s")
    df.set_index("DateTime", inplace=True)
    df.drop("tijdstip", axis=1, inplace=True)
    df.drop("msg_timestamp", axis=1, inplace=True)

    # Combine other namings of raam_id and replace them by their classroom (from lokalen_list)
    df["raam_id"] = df["raam_id"].str.lower()
    df = df[df["raam_id"].str.contains('|'.join(lokalen_list), case=False)]
    to_replace_list = sorted(df["raam_id"].unique().tolist())
    replacement_list = sorted(lokalen_list)
    replacement_list = [
        (matches := difflib.get_close_matches(value.lower(), replacement_list)) and matches[0] or value
        for value in to_replace_list
    ]
    mapping_dict = dict(zip(to_replace_list, replacement_list))
    df["raam_id"] = df["raam_id"].replace(mapping_dict, regex=True)
    
    # Group by and split into a dictionary of DataFrames
    dfs_dict = {key: group for key, group in df.groupby("raam_id")}

    # Find the common DateTime indices window
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values())
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values())
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    for key, df in dfs_dict.items():
        # Aggregate multiple sensors on the same DateTime 
        # Use max() because if at least 1 window is open, raamopen should be 1 (if all closed: 0)
        grouped = df.groupby(df.index)
        df = grouped.max(numeric_only=True)
        # Crop DataFrame to be in common indices range
        df = df.loc[(df.index >= start_datetime) & (df.index <= end_datetime)]
        # Only select necessary columns before downsampling and imputation 
        df = df[selected_features]
        # Skip a few rows until we get to first minutes divisible by 10
        first_divisible_index = df.index[df.index.minute % 10 == 0].min()
        # Downsample to values each 10 minutes by max pooling, also ignore NA or NaN values in max computation
        df = df.loc[first_divisible_index:].resample(rule="10T").max() # max() because if at least 1 window is open, raamopen should be 1
        # Single Imputation by most frequent substitution, since raamopen is a boolean value of 0 or 1
        imputer = SimpleImputer(strategy="most_frequent")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        # Mutate the DataFrame in the dictionary
        dfs_dict[key] = df

    # Find the common DateTime indices window: again after resampling everything
    smallest_datetimes = {key: df.index.min() for key, df in dfs_dict.items()}
    largest_of_smallest_datetimes = max(smallest_datetimes.values()).to_datetime64()
    largest_datetimes = {key: df.index.max() for key, df in dfs_dict.items()}
    smallest_of_largest_datetimes = min(largest_datetimes.values()).to_datetime64()
    start_datetime = largest_of_smallest_datetimes
    end_datetime = smallest_of_largest_datetimes

    # Rename all columns so they become unique per table before merging (to avoid duplicate column names)
    tablename = "ramen"
    for key, df in dfs_dict.items():
        df.columns = [f"{tablename}_{col}" for col in df.columns]
        dfs_dict[key] = df
    
    return dfs_dict, start_datetime, end_datetime


def main() -> None:
    # TODO: might do imputation when all tables are merged per classroom
    # TODO: check if chosen imputation scheme for each table is correct
    # TODO: update data to also include march 2024 fully (do this in april)
    # ------------------------------------------------------------------------------------------------------------------
    # Lokalen
    # ------------------------------------------------------------------------------------------------------------------
    lokalen_list = ["f106", "f107", "f205", "f207"]

    dfs_dict_weather_wsc11, start_weather_wsc11, end_weather_wsc11 = dfs_dict_weather_wsc11_cleanup(selected_features=["rAirTemperature", 
        "rHousingTemperature", "rGlobalRadiation", "rBrightnessEast", "bPrecipitation"], lokalen_list=lokalen_list)
    dfs_dict_radiatoren, start_radiatoren, end_radiatoren = dfs_dict_radiatoren_cleanup(selected_features=["delta_t"], lokalen_list=lokalen_list)
    dfs_dict_ventilatie, start_ventilatie, end_ventilatie = dfs_dict_ventilatie_cleanup(selected_features=["aanvoersnelheid", "afvoersnelheid"], lokalen_list=lokalen_list)
    dfs_dict_klimaat, start_klimaat, end_klimaat = dfs_dict_klimaat_cleanup(selected_features=["temperatuur", "relatieve_vochtigheid", "co2"], lokalen_list=lokalen_list)
    dfs_dict_personentellers, start_personentellers, end_personentellers = dfs_dict_personentellers_cleanup(selected_features=["aantal_personen"], lokalen_list=lokalen_list)
    dfs_dict_ramen, start_ramen, end_ramen = dfs_dict_ramen_cleanup(selected_features=["raamopen"], lokalen_list=lokalen_list)
    # Crop all tables in common window
    #common_start_datetime = np.max([start_weather_wsc11, start_radiatoren, start_ventilatie, start_klimaat, start_personentellers, start_ramen])
    #common_end_datetime = np.min([end_weather_wsc11, end_radiatoren, end_ventilatie, end_klimaat, end_personentellers, end_ramen])
    #dfs_dict_weather_wsc11 = {key: df.loc[(df.index >= common_start_datetime) & (df.index <= common_end_datetime)] for key, df in dfs_dict_weather_wsc11.items()}
    #dfs_dict_radiatoren = {key: df.loc[(df.index >= common_start_datetime) & (df.index <= common_end_datetime)] for key, df in dfs_dict_radiatoren.items()}
    #dfs_dict_ventilatie = {key: df.loc[(df.index >= common_start_datetime) & (df.index <= common_end_datetime)] for key, df in dfs_dict_ventilatie.items()}
    #dfs_dict_klimaat = {key: df.loc[(df.index >= common_start_datetime) & (df.index <= common_end_datetime)] for key, df in dfs_dict_klimaat.items()}
    #dfs_dict_personentellers = {key: df.loc[(df.index >= common_start_datetime) & (df.index <= common_end_datetime)] for key, df in dfs_dict_personentellers.items()}
    #dfs_dict_ramen = {key: df.loc[(df.index >= common_start_datetime) & (df.index <= common_end_datetime)] for key, df in dfs_dict_ramen.items()}
    
    #print(common_start_datetime)
    #print(common_end_datetime)
    #exit(0)

    # Merge all tables via inner join on DateTime indexes
    dfs_dict_combined = {}
    for lokaal in lokalen_list:
        df_combined = pd.merge(left=dfs_dict_weather_wsc11[lokaal], right=dfs_dict_radiatoren[lokaal], 
                               left_index=True, right_index=True, how="inner")
        if dfs_dict_ventilatie[lokaal].shape[0] > 0:    
            df_combined = pd.merge(left=df_combined, right=dfs_dict_ventilatie[lokaal], 
                               left_index=True, right_index=True, how="inner")
        df_combined = pd.merge(left=df_combined, right=dfs_dict_klimaat[lokaal], 
                               left_index=True, right_index=True, how="inner")
        df_combined = pd.merge(left=df_combined, right=dfs_dict_personentellers[lokaal], 
                               left_index=True, right_index=True, how="inner")
        df_combined = pd.merge(left=df_combined, right=dfs_dict_ramen[lokaal], 
                               left_index=True, right_index=True, how="inner")
        df_combined.index.names = ["DateTime"]
        dfs_dict_combined[lokaal] = df_combined

    #print(dfs_dict_combined)
    
    # All data
    #for key, df in dfs_dict_combined.items():
    #    df.to_pickle(f"{CLEAN_DATA_PATH}/{key}_all_data_cleaned.pickle")

    # Training set (e.g. f106_train_oct2022_mar2023.pickle)
    for key, df in dfs_dict_combined.items():
        start_datetime = pd.to_datetime("2022-10-01 00:00:00")
        end_datetime = pd.to_datetime("2023-03-31 23:59:59")
        df_train = df.loc[start_datetime:end_datetime]
        df_train.to_pickle(f"{CLEAN_DATA_PATH}/{key}_train_oct2022_mar2023.pickle")

    # Validation set 1 (e.g. f106_valid1_oct2023_dec2023)
    for key, df in dfs_dict_combined.items():
        start_datetime = pd.to_datetime("2023-10-01 00:00:00")
        end_datetime = pd.to_datetime("2023-12-31 23:59:59")
        df_valid_1 = df.loc[start_datetime:end_datetime]
        df_valid_1.to_pickle(f"{CLEAN_DATA_PATH}/{key}_valid1_oct2023_dec2023.pickle")

    # Validation set 2 (e.g. f106_valid2_feb2024_mar2024)
    for key, df in dfs_dict_combined.items():
        start_datetime = pd.to_datetime("2024-02-01 00:00:00")
        end_datetime = pd.to_datetime("2024-03-31 23:59:59")
        df_valid_2 = df.loc[start_datetime:end_datetime]
        df_valid_2.to_pickle(f"{CLEAN_DATA_PATH}/{key}_valid2_feb2024_mar2024.pickle")

    print(f"Succesfully cleaned all data and serialized it into:\n{CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()