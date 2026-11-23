import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Dense, Dropout, Input, Layer, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#Reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
#enable deterministic operations
tf.config.experimental.enable_op_determinism()



@tf.keras.utils.register_keras_serializable(package = "Custom")
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self,inputs):
        # inputs : batch x time_steps x features
        scores = tf.matmul(inputs, inputs, transpose_b=True)

        weights = tf.nn.softmax(scores, axis=-1)
        outputs = tf.matmul(weights, inputs)
        # return average by rows
        return tf.reduce_mean(outputs, axis = 1)


#Extract degradation features from the health index: HI level, HI slope(how quickly index changes), HI variability(whether sensor becomes noisier)
def add_hi_features(df_master, sensor):
    df = df_master.copy()
    hi = f'HI_{sensor}'
    df_grouped = df.groupby('unit_number')[hi]

    hi_mean_5 = f'hi_mean_5_{sensor}'
    df[hi_mean_5] = df_grouped.transform(lambda s: s.rolling(5, min_periods = 1).mean())

    hi_slope_5 = f'hi_slope_5_{sensor}'
    df[hi_slope_5] = df_grouped.transform( lambda s: (s- s.shift(5))/5)

    hi_std_10 = f'hi_std_10_{sensor}'
    df[hi_std_10] = df_grouped.transform(lambda x: x.rolling(10, min_periods = 2).std())

    df[hi_mean_5]= df[hi_mean_5].fillna(0)
    df[hi_slope_5] = df[hi_slope_5].fillna(0)
    df[hi_std_10]= df[hi_std_10].fillna(0)

    hi_extracts = [hi_mean_5, hi_slope_5, hi_std_10]

    df_master = pd.concat([df_master, df[hi_extracts]],axis = 1)

    return df_master


def extract_sensor_signals(train_df, test_df, columns, prefix, n_components = 1):

    pca_features_train = pd.DataFrame(index=train_df.index)
    pca_features_test = pd.DataFrame(index=test_df.index)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[columns])
    test_scaled = scaler.transform(test_df[columns])

    pca = PCA(n_components = n_components, svd_solver = 'full')


    train_components = pca.fit_transform(train_scaled)
    test_components = pca.transform(test_scaled)

    for i in range(n_components):
        pca_feature_name = f'{prefix}_PCA_{i+1}'
        pca_features_train[pca_feature_name] = train_components[:,i]
        pca_features_test[pca_feature_name] = test_components[:,i]
        pca_features.append(pca_feature_name)


    train_df = pd.concat([train_df, pca_features_train], axis=1)
    test_df = pd.concat([test_df, pca_features_test], axis=1)

    return train_df, test_df


def fit_cluster_normalize(train_df):
    cluster_settings = (train_df['operational_setting_1'].astype(str) + '_' + train_df['operational_setting_2'].astype(str))

    stats = {}

    for feature in feature_sensors:
        stats[feature] = train_df.groupby(cluster_settings)[feature].agg(['std','mean'])


    return stats

def apply_cluster_normalize(df, stats):
    df = df.copy()

    cluster_settings = (train_df['operational_setting_1'].astype(str) + '_' + train_df['operational_setting_2'].astype(str))


    for feature in feature_sensors:
        mean = cluster_settings.map(stats[feature]['mean'])
        std = cluster_settings.map(stats[feature]['std'])

        mean = mean.fillna(stats[feature]['mean'].mean())
        std = std.fillna(stats[feature]['std'].median())

        std = std.replace(0, 1)

        df[feature] = (df[feature] - mean) /(std + 1e-4)

    return df

def ema(df):
    #exponential moving average
    df[feature_sensors] = df.groupby('unit_number')[feature_sensors].transform(lambda x: x.ewm(span = 10, adjust= False ).mean())
    return df



def create_sequences(df, feature_cols, window_size):
    X_seq, y_seq, groups = [], [], []
    #30 cycle sliding window per engine unit
    for unit, group in df.groupby('unit_number'):
        group_features = group[feature_cols].values
        group_targets = group['rul'].values

        for i in range(window_size, len(group)):
            X_seq.append(group_features[i - window_size:i])
            y_seq.append(group_targets[i])
            groups.append(unit)



    return np.array(X_seq), np.array(y_seq), np.array(groups)


#register custom loss function with Keras serialization
@tf.keras.utils.register_keras_serializable(package="CMAPSS",name="cmapss_scoring_function")
def cmapss_scoring_function(y_true,y_pred):

    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    d = y_pred - y_true
    under_estimation_score = tf.exp(-d/13.0) - 1.0
    over_estimation_score = tf.exp(d/10.0) - 1.0
    penalty_score = tf.where(d<0, under_estimation_score, over_estimation_score)

    return tf.reduce_mean(penalty_score)

def cmapss_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    for x in [y_true, y_pred]:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = np.asarray(x)

    # tf broadcasting: reshape to (batch_size, 1) tensors
    y_pred = tf.reshape(y_pred, (-1, 1))
    y_true = tf.reshape(y_true, (-1, 1))

    error = y_pred - y_true
    score = tf.where(error<0,tf.exp(-error/13.0) - 1.0, tf.exp(error/10.0) - 1.0)

    return score


rul_file = r'./RUL_FD001.txt'
test_file = r'./test_FD001.txt'
train_file = r'./train_FD001.txt'


                                    ###############
                                    # RUL data file
                                    ###############


rul_dict = {"RUL": []}
f_length = 0

with open(rul_file, 'r') as f:

    for line in f.readlines():
        f_length += 1
        line = line.strip()
        line = line.split()
        for k,v in zip(rul_dict.keys(), line):
            rul_dict[k].append(v)



rul_df = pd.DataFrame(rul_dict)
rul_df["unit_number"] = [k for k in range(1,f_length+1)]
rul_df= rul_df.reindex(columns = ["unit_number","RUL"])
rul_df = rul_df.astype(int)


                                    ################
                                    # Test Data File
                                    ################


test_header = [
                "unit_number", "cycle_time", "operational_setting_1","operational_setting_2","operational_setting_3",
                "sensor_measurement_1","sensor_measurement_2","sensor_measurement_3","sensor_measurement_4",
                "sensor_measurement_5","sensor_measurement_6","sensor_measurement_7","sensor_measurement_8",
                "sensor_measurement_9","sensor_measurement_10","sensor_measurement_11","sensor_measurement_12",
                "sensor_measurement_13","sensor_measurement_14","sensor_measurement_15","sensor_measurement_16",
                "sensor_measurement_17","sensor_measurement_18","sensor_measurement_19","sensor_measurement_20",
                "sensor_measurement_21"
            ]
sensor_measurements = ["sensor_measurement_1","sensor_measurement_2","sensor_measurement_3","sensor_measurement_4",
                "sensor_measurement_5","sensor_measurement_6","sensor_measurement_7","sensor_measurement_8",
                "sensor_measurement_9","sensor_measurement_10","sensor_measurement_11","sensor_measurement_12",
                "sensor_measurement_13","sensor_measurement_14","sensor_measurement_15","sensor_measurement_16",
                "sensor_measurement_17","sensor_measurement_18","sensor_measurement_19","sensor_measurement_20",
                "sensor_measurement_21"]

stable_sensors= ["sensor_measurement_1","sensor_measurement_5","sensor_measurement_6","sensor_measurement_10",
                 "sensor_measurement_16","sensor_measurement_18","sensor_measurement_19"]

feature_sensors = [x for x in sensor_measurements if x not in stable_sensors]

# removing stable sensors from physical/virtual sensors
temperature_sensors = [x for x in ["sensor_measurement_1","sensor_measurement_2","sensor_measurement_3","sensor_measurement_4"] if x not in stable_sensors]
pressure_sensors = [x for x in ["sensor_measurement_5","sensor_measurement_6","sensor_measurement_7","sensor_measurement_11"]if x not in stable_sensors]
speed_sensors = [x for x in ["sensor_measurement_8","sensor_measurement_9","sensor_measurement_13","sensor_measurement_14","sensor_measurement_18","sensor_measurement_19"]if x not in stable_sensors]
ratio_sensors = [x for x in ["sensor_measurement_10","sensor_measurement_12","sensor_measurement_15","sensor_measurement_16"]if x not in stable_sensors]
energy_sensors = [x for x in ["sensor_measurement_17","sensor_measurement_20","sensor_measurement_21"]if x not in stable_sensors]


operation_settings = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3']

test_dict = {k:[] for k in test_header}

with open(test_file, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = line.split()

        for k,v in zip(test_header, line):
                test_dict[k].append(v)

test_df = pd.DataFrame(test_dict)
test_df = test_df.astype(float)
test_df = test_df.astype({
    'unit_number':int,
    'cycle_time' : int
})

engines_n = test_df.groupby('unit_number').ngroups



                                    ####################
                                    # Training Data File
                                    ####################


train_header = test_header

train_dict = {k:[] for k in train_header}

with open(train_file, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = line.split()


        for k,v in zip(train_header, line):
            train_dict[k].append(v)


train_df = pd.DataFrame(train_dict)
train_df = train_df.astype(float)
train_df = train_df.astype({
    'unit_number': int,
    'cycle_time' : int
})



                                        #################################################
                                        # Normalization by cluster and ema smoothing
                                        #################################################

stats = fit_cluster_normalize(train_df)

train_df = ema(apply_cluster_normalize(train_df, stats))
test_df = ema(apply_cluster_normalize(test_df, stats))



                                        ##########################################
                                        # Linear regression: of all noise sensors
                                        ##########################################


# maximum cycle time per engine training trajectory
max_train_cycle = train_df.groupby('unit_number')['cycle_time'].transform('max')

# true training RUL
true_training_rul = max_train_cycle - train_df['cycle_time']
train_df['rul'] = true_training_rul


                                        # Feature engineering

# Correct sensors for operating settings : to separate changes caused by operating conditions from changes potentially caused by degradation.
# fit a regression model to all sensor  (test and train data) during first 30 cycles of every training engine
engineered_features = []
for sensor in feature_sensors:

    test_copy = test_df.copy()
    train_copy = train_df.copy()


    # features X : operation settings
    X_early = train_copy.loc[train_copy['cycle_time'] <= 30,operation_settings]

    #Target y : sensor measurement
    y_early = train_copy.loc[train_copy['cycle_time'] <= 30,sensor]

    #Ridge regression, scale  features using StandardScaler, and chain them using make_pipeline
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha = 0.1))

    pipeline.fit(X_early, y_early)

    sensor_residuals = f'{sensor}_residuals'

    for df in [test_copy, train_copy]:
        #make predictions
        y_predict = pipeline.predict(df[operation_settings])

        df[sensor_residuals] = 0.0
        n = min(len(y_predict), len(df))
        df.iloc[:n, df.columns.get_loc(sensor_residuals)] = (df[sensor].iloc[:n].to_numpy() - y_predict[:n])

    # early 30 cycles in training data as a reference
    train_cycle_early = train_copy['cycle_time'] <= 30

    early_train_residual = train_copy.loc[train_cycle_early, sensor_residuals]
    early_train_residual_mean = early_train_residual.mean()
    early_train_residual_std = early_train_residual.std()

    hi_raw = f'HI_Raw_{sensor}'
    hi = f'HI_{sensor}'

    for df in [test_copy, train_copy]:
        #Raw Health index definition for sensors: normalized to early training data
        df[hi_raw] = (df[sensor_residuals] - early_train_residual_mean) / early_train_residual_std

    #orient health index towards degradation
    correlation = train_df['rul'].corr(train_copy[hi_raw])
    direction = -1 if correlation > 0 else 1

    test_copy[hi] = direction * test_copy[hi_raw]
    train_copy[hi] = direction * train_copy[hi_raw]

    health_columns = [hi_raw, hi]
    test_df = pd.concat([test_df, test_copy[health_columns]], axis = 1)
    train_df = pd.concat([train_df, train_copy[health_columns]], axis = 1)


    test_df= add_hi_features(test_df, sensor)
    train_df= add_hi_features(train_df, sensor)
    engineered_features.extend([f'HI_{sensor}',f'hi_mean_5_{sensor}', f'hi_slope_5_{sensor}',f'hi_std_10_{sensor}'])

# PCA



temperature_cols = temperature_sensors + [feature  for sensor in temperature_sensors for feature in (f'HI_{sensor}', f'hi_mean_5_{sensor}', f'hi_slope_5_{sensor}', f'hi_std_10_{sensor}')]
pressure_cols = pressure_sensors + [feature  for sensor in pressure_sensors for feature in  (f'HI_{sensor}', f'hi_mean_5_{sensor}', f'hi_slope_5_{sensor}', f'hi_std_10_{sensor}') ]
speed_cols = speed_sensors + [feature for sensor in speed_sensors for feature in (f'HI_{sensor}', f'hi_mean_5_{sensor}', f'hi_slope_5_{sensor}', f'hi_std_10_{sensor}')]
ratio_cols = ratio_sensors + [feature for sensor in ratio_sensors for feature in (f'HI_{sensor}', f'hi_mean_5_{sensor}', f'hi_slope_5_{sensor}', f'hi_std_10_{sensor}')]
energy_cols = energy_sensors + [feature for sensor in energy_sensors for feature in (f'HI_{sensor}', f'hi_mean_5_{sensor}', f'hi_slope_5_{sensor}', f'hi_std_10_{sensor}')]

# separate PCA for each physical category.

n_components = 1
pca_features = []

for feature_column, prefix in [(temperature_cols, 'temperature'),(pressure_cols, 'pressure'),(speed_cols, 'speed'),(ratio_cols, 'ratio'),(energy_cols, 'energy')]:
    train_df, test_df = extract_sensor_signals(train_df, test_df, feature_column, prefix, n_components)



                                        #################################
                                        #degradation and RUL relationship
                                        #################################


#predictors
features = [
    'operational_setting_1',
    'operational_setting_2',
    'cycle_time'
]


#PCA features
features = features + pca_features


                                        # Models

#Sequential State Space Model
window_size = 30
min_max_train_rul = train_df.groupby('unit_number')['rul'].max().min()
MAX_RUL = 0.98 * min_max_train_rul

X_seq_3d, y_seq_3d, groups = create_sequences(train_df, features, window_size)
y_seq_3d = np.minimum(y_seq_3d, MAX_RUL)

#BI LSTM

model = Sequential([
    Input(shape = (window_size, len(features))),
    #learns temporal patterns and representations.
    Bidirectional(LSTM(64, return_sequences = True)),
    Dropout(0.2),
    #learns Higher level temporal patterns and representations.
    Bidirectional(LSTM(32, return_sequences = True)),
    Dropout(0.2),
    # identifies which time steps are important for prediction
    #explicitly learn a weighted combination of all 30 time steps.
    AttentionLayer(),
    # learns non linear relationships before final prediction
    Dense(32, activation = 'relu'),
    Dropout(0.1),
    Dense(1, activation = 'linear')
])



                                            #Model Fitting

#lstm_pipeline = model
#lstm_pipeline.fit(X_seq_3d, y_seq_3d, batch_size = 64, epochs = 500, callbacks= [early_stopping, checkpoint], validation_split = 0.1, shuffle= False)

lstm_pipeline = keras.models.load_model('state_space_model_bilstm_30_ema.keras')


                                                    # Models Predictions

# at every cycle time of test data a rul is predicted,
# therefore the rul predicted at the end of each test trajectory is what to be compared with true test rul from rul data
# prediction last 30 cycles
prediction_range = 30

                                                # prediction on engine 14

engine = 8
test_unit  = test_df[test_df['unit_number']==engine]

test_unit_cutoff = np.array(test_unit[features])
test_unit_cutoff = test_unit_cutoff[-prediction_range:]
test_unit_cutoff = np.expand_dims(test_unit_cutoff, axis = 0)

test_unit_rul_predicted = lstm_pipeline.predict(test_unit_cutoff)

#test predicted rul at last trajectories cycle
last_cycle_rul_predicted = test_unit_rul_predicted[-1]

print(f'Model prediction on test engine number {engine}')
print('True RUL: ', rul_df.loc[rul_df['unit_number']==engine, "RUL"].iloc[0], f'vs RUL Predicted: {last_cycle_rul_predicted[0]:.2f}')

mae = mean_absolute_error(rul_df[rul_df['unit_number']==engine]["RUL"], [last_cycle_rul_predicted])
mse = mean_squared_error(rul_df[rul_df['unit_number']==engine]["RUL"], [last_cycle_rul_predicted])
rmse = np.sqrt(mse)

cmapss_scoring = cmapss_scoring_function(rul_df[rul_df['unit_number'] == engine]["RUL"], last_cycle_rul_predicted)
cmapss_scoring = cmapss_scoring.numpy()

print(f'Model prediction on test engine number {engine}: performance metrics')
print(f'mae:{mae:.2f} | mse: {mse:.2f} | rmse: {rmse:.2f} | C-MAPSS Scoring:  {cmapss_scoring:.2f}')


                                                # prediction on all engines


# test trajectories last indices
last_index= test_df.groupby('unit_number')['cycle_time'].idxmax()

test_engines_cutoff = np.array(test_df.groupby('unit_number')[features].tail(prediction_range))

test_engines_cutoff = test_engines_cutoff.reshape(engines_n, prediction_range, len(features))


test_rul_predicted = lstm_pipeline.predict(test_engines_cutoff)

last_rul_predicted = test_rul_predicted
last_rul_predicted = pd.DataFrame(last_rul_predicted)

mae = mean_absolute_error(rul_df["RUL"], last_rul_predicted)
mse = mean_squared_error(rul_df["RUL"], last_rul_predicted)
rmse = np.sqrt(mse)
r2s = r2_score(rul_df["RUL"], last_rul_predicted)
cmapss_scoring = cmapss_scoring_function(rul_df["RUL"], last_rul_predicted)
cmapss_scoring = cmapss_scoring.numpy()

score = cmapss_score(rul_df["RUL"], last_rul_predicted)

print('Model prediction on all engines: performance metrics')
print(f'mae:{mae:.2f} | mse: {mse:.2f} | rmse: {rmse:.2f} | r2:  {r2s:.2f} | C-MAPSS Scoring:  {cmapss_scoring:.2f}')



error = last_rul_predicted.sub(rul_df["RUL"], axis = 0)

score_df = pd.DataFrame(score)
plt.scatter(error, score_df)
plt.show()

