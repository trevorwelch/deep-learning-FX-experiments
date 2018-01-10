# Data prep and loading function
# Need to break this apart into smaller helper functions

def prepare_data(ohlc_data,  
                 lookforward, 
                 lookback,
                 feature_generation_channels,
                 lag=1,
                 change_open_close=1, 
                 range_high_low=1,
                 rolling_mean_range_HL=1,
                 range_high_close=1,
                 std_dev=1,
                 rolling_mean=1,
                 change_open_close_shift=0,
                 range_high_low_shift=0,
                 range_high_close_shift=0,
                 diff_shift=0,
                 rolling_mean_shift=0,
                 std_dev_shift=0,
                 center=False
                ):

    
    # Make sure df_ohlc_and_rectangles is an empty df
    df_ohlc_and_rectangles = pd.DataFrame(data=None)
    
    # Instantiate X as an empty df
    X = pd.DataFrame(data=None) 
        
    # Instantiate y as an empty df
    y = pd.DataFrame(data=None) 
    
    # Instantiate dict of scalers to save scalers for later
    dict_of_scalers = {}
    
    # Initialize groups list for GroupKFold
    groups = []
    
    # Initialize list of 
    csv_tags = []

    # Initialize groups counter for later use in GroupKFold
    groups_count = 0
    
    # Initialize groups_dict for storing CSV : group count pairs
    groups_dict = {}
    
    # Loop to pull in parse and clean up the data 
    for index, each in enumerate(ohlc_data):
        
        print("Processing: ", each, "Number ", index+1, " of ", len(ohlc_data))
        print("We now have:", df_ohlc_and_rectangles.shape, "rows of data!")
        # load ohlc, label columns
        print("Reading in Date, OHLCV, Supply/Demand")
        df_ohlc = pd.read_csv(each, usecols=['Date', 'O', 'H', 'L', 'C', 'V', 'Supply/Demand'])
        print("Renaming df_ohlc column headers for clarity")
        df_ohlc.columns = ['Epoch Date', 'O', 'H', 'L', 'C', 'V', 'Supply/Demand']
        
        # load rects, label columns    
        print("Reading in Rectangle data")
        df_rects = pd.read_csv(each, usecols=['Object Name', 'Date Start', 'Date End', 'Proximal', 'Distal']) 
        print("Renaming df_rects column headers for clarity")
        df_rects.columns = ['Object Name', 'Epoch Date Start', 'Epoch Date End', 'Proximal', 'Distal']
        # drop nans from df_rectangles
        df_rects = df_rects.dropna() 
        
        # Merge the data into one to do our identifications
        df_merged_unscaled = df_ohlc.merge(df_rects, left_on='Epoch Date', right_on='Epoch Date Start', how='left')
        
        # run supply or demand zone identification function, which adds Supply and Demand markers
        print("Identifying supply/demand with supply_or_demand func:", each, df_merged_unscaled.shape)
        print("~~~")
        supply_or_demand(df_merged_unscaled)
        
        # Identify the ends of tagged zones, later on we will have to use this for predicted zones as well
        print("Identifying zone ends with zone_ender func:", each, df_merged_unscaled.shape)
        print("~~~")        
        zone_ender(df_merged_unscaled)
        

        # Create classes that describe where proximal/distal match in the data
        channels = ['O', 'H', 'L', 'C']
        df_merged_unscaled_with_labels, channels = proximal_distal_class_matching(df_merged_unscaled, 
                                                                      channels, 6, 6)
        
        distal_list = [col for col in list(df_merged_unscaled_with_labels) if col.startswith('Distal')]
        #print("list of distal columns:", distal_list)
        
        proximal_list = [col for col in list(df_merged_unscaled_with_labels) if col.startswith('Proximal')]
        #print("list of proximal columns:", proximal_list)
        
        # Create separate df for the data we want to be normalized
        df_norm = df_merged_unscaled_with_labels[['O', 
                                      'H', 
                                      'L', 
                                      'C',
                                        'V',
#                                       'Proximal', 
#                                       'Distal'
                                     ]]
        print("Your df_norm columns:", df_norm.columns)

        # Create separate df of the unscaled data
        df_non_norm = df_merged_unscaled_with_labels.drop(
            ['O', 
            'H', 
            'L', 
            'C',
             'V',
            #'Proximal', 
            #'Distal'
            ],axis=1)
        print("Your df_non_norm columns:", df_non_norm.columns)

        if lag == 1:
            df_norm = lag_data(df_norm)

        # Generate features *before scaling*, including a lot of new columns from different data. 
        df_features = generate_features(
                                        df_norm, 
                                        lookforward, 
                                        lookback, 
                                        feature_generation_channels,
                                        change_open_close=change_open_close, 
                                        range_high_low=range_high_low,
                                        rolling_mean_range_HL=rolling_mean_range_HL,
                                        range_high_close=range_high_close,
                                        std_dev=std_dev,
                                        rolling_mean=rolling_mean,
                                        change_open_close_shift=change_open_close_shift,
                                        range_high_low_shift=range_high_low_shift,
                                        range_high_close_shift=range_high_close_shift,
                                        diff_shift=diff_shift,
                                        rolling_mean_shift=rolling_mean_shift,
                                        std_dev_shift=std_dev_shift,
                                        center=center
        )
        #print("df_features is:", df_features.columns) 
        #print("df_norm columns is:", df_norm.columns)

        # Merge our generated features back together with our data
        df_norm_and_features = pd.concat([df_norm, df_features], axis=1)
        #print("df_norm_and_features shape is:", df_norm_and_features.shape)
        
        # Create a list of all the pre-scaled columns 
        df_imputed_norm_and_features_columns = list(df_norm_and_features)
        
        # Impute NaNs so we don't confuse lame algorithms who won't behave with them later on
        
        fill_NaN = Imputer(strategy='mean', axis=0)
        df_imputed_norm_and_features = pd.DataFrame(fill_NaN.fit_transform(df_norm_and_features))
        

        
        # Scale the data #
        
        # Clear the scaler var just in case
        scaler = None
        # Create scaler object
        scaler = StandardScaler()
        # Scale the dataframe
        df_scale = pd.DataFrame(scaler.fit_transform(df_imputed_norm_and_features.values))
        
        # Set current_scaler to the scaler we just used
        current_scaler = scaler
        
        # Add to a dict the scaler we just used
        dict_of_scalers[each] = current_scaler
        
        # Additional features, PCA decomposition

        # Instantiate PCA 
        pca = PCA(n_components=12)
        
        # Fit transform the PCA into new df of only PCA features
        df_X_PCA = pd.DataFrame(pca.fit_transform(df_scale))
        df_X_PCA.columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10', 'PCA11', 'PCA12']
        
        # Create a list of all the PCA_columns (basically just a list of numbers for n of PCA features)
        df_X_PCA_columns = list(df_X_PCA)
        
        # Add the scaled data together with the PCA data
        df_scale_with_PCA = pd.concat([df_scale.reset_index(drop=True), df_X_PCA.reset_index(drop=True)], axis=1)
        print("Your df_scale_with_PCA:", list(df_scale_with_PCA))
        
        # Add on to the end of the df_imputed_norm_and_features_columns list the df_X_PCA_columns list
        df_imputed_norm_and_features_columns.extend(df_X_PCA_columns)
        
        #print(df_imputed_norm_and_features_columns)
        
        # Name the columns of the scaled data + PCA df columns in order
        df_scale_with_PCA.columns = df_imputed_norm_and_features_columns
        print("Your df_scale_with_PCA columns are:", list(df_scale_with_PCA))

        df_imputed_norm = df_scale_with_PCA
       

        #print("df_imputed_norm_and_features columns are:", df_imputed_norm.columns) 

        #print(df_non_norm.columns)
        df_merged = pd.concat([df_imputed_norm, df_non_norm], axis=1)
        print("Merging the scaled data and the unscalable data back together with pd.concat:", each, df_merged.shape)

        # Add to our groups list so that later we know how many samples came from where
        #print(each, " contains ", len(df_merged_and_features), " samples.")
        #print("Adding length of ", each, " to groups.") 
        groups.extend([groups_count] * len(df_merged))
        csv_tags.extend([each] * len(df_merged))
        
            
            
        # Add to a dict the scaler we just used
        groups_dict[each] = groups_count
                      
        groups_count +=1
        
        # Add the current dataframe to the bottom of the master dataframe
        df_ohlc_and_rectangles = pd.concat([df_merged, df_ohlc_and_rectangles])

        print("Finished:", each, df_merged.shape)
        print("--------------------------------")

    
    X1 = df_ohlc_and_rectangles.drop(distal_list, axis=1)
    X = X1.drop(proximal_list, axis=1)
    
    df_supply_demand = df_ohlc_and_rectangles[['rectangle_here']].fillna(False)
    y1 = pd.concat([df_ohlc_and_rectangles[distal_list], df_ohlc_and_rectangles[proximal_list]], axis=1)
    y2 = pd.concat([df_supply_demand, y1 ], axis=1)
    y = y2.drop(['Distal', 'Proximal'], axis=1).astype(int)
    
    
    # First make the list of csv_tags into a Series:
    csv_tags_series = pd.Series(csv_tags)
    # Then add the values to the DataFrame
    df_ohlc_and_rectangles['csv_origin_tag'] = csv_tags_series.values
    
    # First make the list of groups into a Series:
    groups_series = pd.Series(groups)
    # Then add the values to the DataFrame
    df_ohlc_and_rectangles['group'] = groups_series.values
    X['group'] = groups_series.values
    

    
    print("Finished loading data! You'll probably still want to remove some columns though")
    return X, y, df_ohlc_and_rectangles, np.array(groups), groups_count, dict_of_scalers, groups_dict, df_imputed_norm_and_features_columns