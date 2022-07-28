import datetime
import keras_tuner as kt
import numpy as np
import os
import pandas as pd
import pdb
import time

from keras.activations import relu
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, LSTM
from keras.metrics import RootMeanSquaredError, Accuracy
from keras.models import Sequential
from keras.regularizers import L1, L2, L1L2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.optimizers import Adam

# Changes directory to src
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from utils import *

# Change back to project root directory
os.chdir("../")

sk_holidays = []

df = pd.read_csv(
    "data/intermediate/df_00.csv"
).drop(
    "date", 
    axis=1
)

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.set_index("Datetime")

whole_days = df.groupby(df.index.date).size()==24

whole_days_index = whole_days.index[whole_days.tolist()]

df = df[df.index.floor("d").isin(whole_days_index)]

sk_holidays = [
    "2016-11-25", 
    "2017-01-01", 
    "2017-01-27", 
    "2017-01-28", 
    "2017-01-29", 
    "2017-01-30"
]

df["holiday"] = df.index.floor("d").isin(sk_holidays)
df["holiday"] = df["holiday"].apply(lambda x: int(x))

df = EncodeCyclical(df, "hour")
df = EncodeCyclical(df, "day")
df = EncodeCyclical(df, "month")

df = CreateLags(df, "ap_total", 5)
df = CreateLags(df, "ap_TV", 5)
df = CreateLags(df, "ap_washing-machine", 5)
df = CreateLags(df, "ap_rice-cooker", 5)
df = CreateLags(df, "ap_water-purifier", 5)
df = CreateLags(df, "ap_microwave", 5)
df = CreateLags(df, "ap_kimchi-fridge", 5)

# Drop indicator variables
df = df[df.columns.drop(list(df.filter(regex="i_")))]

encoder = OneHotEncoder(sparse=False)

time_features = [
    "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month","holiday"
]

df_cats = df.loc[:,df.columns[df.dtypes=="object"]]

df_cats_arr = encoder.fit_transform(df_cats)

df_num_arr = df.loc[:,df.columns[df.dtypes!="object"]].values

time_features_colnum = [
    i for i, x in enumerate(
        df.columns[df.dtypes!="object"].isin(time_features)
    ) if x
]

df_arr = np.append(df_num_arr, df_cats_arr, axis=1)

y = df_arr[:,0]

X = df_arr[:,1:]

time_features_colnum = [int(i)-1 for i in time_features_colnum]

# Get number of numeric columns in X
num_num_cols = X.shape[1]-df_cats_arr.shape[1]+1

num_cols_excl_time = RemoveItem(time_features_colnum, RangeList(num_num_cols))

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    train_size=round((df_arr.shape[0]/24)*0.8)*24,
    test_size=round((df_arr.shape[0]/24)*(1-0.8))*24,
    shuffle=False
)

# Feature scaler
f_scaler = MinMaxScaler()
X_train[
    :, num_cols_excl_time
] = f_scaler.fit_transform(
    X_train[
        :, 
        num_cols_excl_time
    ]
)

# Only need to transform test dataset because scaling parameters from training dataset are used
X_test[
    :, num_cols_excl_time
] = f_scaler.transform(
    X_test[
        :, 
        num_cols_excl_time
    ]
)# Target scaler
t_scaler = MinMaxScaler()
y_train = t_scaler.fit_transform(y_train.reshape(-1, 1))
# Only need to transform test dataset because scaling parameters from training dataset are used
y_test = t_scaler.transform(y_test.reshape(-1, 1))

# Reshape input -> 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

def build_model(
    hp,
    X_train=X_train,
    y_train=y_train,
):
    
    num_lstm_layers = hp.Int(
        "num_lstm_layers",
        min_value=1,
        max_value=4,
        step=1
    )

    print("Building model...")
    model = Sequential()
    print("Adding first LSTM layer...")
    lstm_1_units = hp.Int(
        "lstm_1_input_unit", 
        min_value=16, 
        max_value=512, 
        step=16
    )
    kernel_reg_lstm_1 = RegWrapper(
        hp.Choice("kernel_lstm_1_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "kernel_lstm_1_reg_value", 
            min_value=0, 
            max_value=0.5, 
            step=0.01
        )
    )
    bias_reg_lstm_1 = RegWrapper(
        hp.Choice("bias_lstm_1_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float("bias_lstm_1_reg_value", min_value=0, max_value=0.5, step=0.01)
    )
    activity_reg_lstm_1 = RegWrapper(
        hp.Choice("activity_lstm_1_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "activity_lstm_1_reg_value", 
            min_value=0, 
            max_value=0.5, 
            step=0.01
        )
    )
    recurrent_reg_lstm_1 = RegWrapper(
        hp.Choice("recurrent_lstm_1_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "recurrent_lstm_1_reg_value", 
            min_value=0, 
            max_value=0.5, 
            step=0.01
        )
    )
    if num_lstm_layers == 1:
        model.add(
            LSTM(
                lstm_1_units, 
                return_sequences=False,
                kernel_regularizer=kernel_reg_lstm_1,
                bias_regularizer=bias_reg_lstm_1,
                activity_regularizer=activity_reg_lstm_1,
                recurrent_regularizer=recurrent_reg_lstm_1,
                dropout=hp.Float(
                    "lstm_1_dropout_rate", 
                    min_value=0.0, 
                    max_value=0.8, 
                    step=0.1
                )
            )
        )
    if num_lstm_layers > 1:
        model.add(
            LSTM(
                lstm_1_units, 
                return_sequences=True,
                kernel_regularizer=kernel_reg_lstm_1,
                bias_regularizer=bias_reg_lstm_1,
                activity_regularizer=activity_reg_lstm_1,
                recurrent_regularizer=recurrent_reg_lstm_1,
                dropout=hp.Float(
                    "lstm_1_dropout_rate", 
                    min_value=0.0, 
                    max_value=0.8, 
                    step=0.1
                )
            )
        )
        lstm_2_units = hp.Int(
            "lstm_2_input_unit", 
            min_value=16, 
            max_value=512, 
            step=16
        )
        kernel_reg_lstm_2 = RegWrapper(
            hp.Choice("kernel_lstm_2_reg_type", ["l1", "l2", "l1l2"]), 
            hp.Float(
                "kernel_lstm_2_reg_value", 
                min_value=0, 
                max_value=0.5, 
                step=0.01
            )
        )
        bias_reg_lstm_2 = RegWrapper(
            hp.Choice("bias_lstm_2_reg_type", ["l1", "l2", "l1l2"]), 
            hp.Float(
                "bias_lstm_2_reg_value", 
                min_value=0, 
                max_value=0.5, 
                step=0.01)
        )
        activity_reg_lstm_2 = RegWrapper(
            hp.Choice("activity_lstm_2_reg_type", ["l1", "l2", "l1l2"]), 
            hp.Float(
                "activity_lstm_2_reg_value", 
                min_value=0, 
                max_value=0.5, 
                step=0.01
            )
        )
        recurrent_reg_lstm_2 = RegWrapper(
            hp.Choice("recurrent_lstm_2_reg_type", ["l1", "l2", "l1l2"]), 
            hp.Float(
                "recurrent_lstm_2_reg_value", 
                min_value=0, 
                max_value=0.5, 
                step=0.01
            )
        )
        print("Adding second LSTM layer...") 
        if num_lstm_layers==2:
            model.add(
                LSTM(
                    lstm_2_units, 
                    return_sequences=False,
                    kernel_regularizer=kernel_reg_lstm_2,
                    bias_regularizer=bias_reg_lstm_2,
                    activity_regularizer=activity_reg_lstm_2,
                    recurrent_regularizer=recurrent_reg_lstm_2,
                    dropout=hp.Float(
                        "lstm_2_dropout_rate", 
                        min_value=0.0, 
                        max_value=0.8, 
                        step=0.1
                    )
                )
            )
        if num_lstm_layers>2:
            model.add(
                LSTM(
                    lstm_2_units, 
                    return_sequences=True,
                    kernel_regularizer=kernel_reg_lstm_2,
                    bias_regularizer=bias_reg_lstm_2,
                    activity_regularizer=activity_reg_lstm_2,
                    recurrent_regularizer=recurrent_reg_lstm_2,
                    dropout=hp.Float(
                        "lstm_2_dropout_rate", 
                        min_value=0.0, 
                        max_value=0.8, 
                        step=0.1
                    )
                )
            )
            print("Adding third LSTM layer...")
            lstm_3_units = hp.Int(
                "lstm_3_input_unit", 
                min_value=16, 
                max_value=512, 
                step=16
            )
            kernel_reg_lstm_3 = RegWrapper(
                hp.Choice("kernel_lstm_3_reg_type", ["l1", "l2", "l1l2"]), 
                hp.Float(
                    "kernel_lstm_3_reg_value", 
                    min_value=0, 
                    max_value=0.5, 
                    step=0.01
                )
            )
            bias_reg_lstm_3 = RegWrapper(
                hp.Choice("bias_lstm_3_reg_type", ["l1", "l2", "l1l2"]), 
                hp.Float(
                    "bias_lstm_3_reg_value", 
                    min_value=0, 
                    max_value=0.5, 
                    step=0.01
                )
            )
            activity_reg_lstm_3 = RegWrapper(
                hp.Choice("activity_lstm_3_reg_type", ["l1", "l2", "l1l2"]), 
                hp.Float(
                    "activity_lstm_3_reg_value", 
                    min_value=0, 
                    max_value=0.5, 
                    step=0.01
                )
            )
            recurrent_reg_lstm_3 = RegWrapper(
                hp.Choice("recurrent_lstm_3_reg_type", ["l1", "l2", "l1l2"]), 
                hp.Float(
                    "recurrent_lstm_3_reg_value", 
                    min_value=0, 
                    max_value=0.5, 
                    step=0.01
                )
            )
            if num_lstm_layers==3:
                model.add(
                    LSTM(
                        lstm_3_units, 
                        return_sequences=False,
                        kernel_regularizer=kernel_reg_lstm_3,
                        bias_regularizer=bias_reg_lstm_3,
                        activity_regularizer=activity_reg_lstm_3,
                        recurrent_regularizer=recurrent_reg_lstm_3,
                        dropout=hp.Float(
                            "lstm_3_dropout_rate", 
                            min_value=0.0, 
                            max_value=0.8, 
                            step=0.1
                        )
                    )
                )
            if num_lstm_layers>3:
                model.add(
                    LSTM(
                        lstm_3_units, 
                        return_sequences=True,
                        kernel_regularizer=kernel_reg_lstm_3,
                        bias_regularizer=bias_reg_lstm_3,
                        activity_regularizer=activity_reg_lstm_3,
                        recurrent_regularizer=recurrent_reg_lstm_3,
                        dropout=hp.Float(
                            "lstm_3_dropout_rate", 
                            min_value=0.0, 
                            max_value=0.8, 
                            step=0.1
                        )
                    )
                )
                print("Adding fourth LSTM layer...")
                lstm_4_units = hp.Int(
                    "lstm_4_input_unit", 
                    min_value=16, 
                    max_value=512, 
                    step=16
                )
                kernel_reg_lstm_4 = RegWrapper(
                    hp.Choice("kernel_lstm_4_reg_type", ["l1", "l2", "l1l2"]), 
                    hp.Float(
                        "kernel_lstm_4_reg_value", 
                        min_value=0, 
                        max_value=0.5, 
                        step=0.01
                    )
                )
                bias_reg_lstm_4 = RegWrapper(
                    hp.Choice("bias_lstm_4_reg_type", ["l1", "l2", "l1l2"]), 
                    hp.Float(
                        "bias_lstm_4_reg_value", 
                        min_value=0, 
                        max_value=0.5, 
                        step=0.01
                    )
                )
                activity_reg_lstm_4 = RegWrapper(
                    hp.Choice(
                        "activity_lstm_4_reg_type", ["l1", "l2", "l1l2"]
                    ), 
                    hp.Float(
                        "activity_lstm_4_reg_value", 
                        min_value=0, 
                        max_value=0.5, 
                        step=0.01
                    )
                )
                recurrent_reg_lstm_4 = RegWrapper(
                    hp.Choice(
                        "recurrent_lstm_4_reg_type", ["l1", "l2", "l1l2"]
                    ), 
                    hp.Float(
                        "recurrent_lstm_4_reg_value", 
                        min_value=0, 
                        max_value=0.5, 
                        step=0.01
                    )
                )
                if num_lstm_layers==4:
                    model.add(
                        LSTM(
                            lstm_4_units, 
                            return_sequences=False,
                            kernel_regularizer=kernel_reg_lstm_4,
                            bias_regularizer=bias_reg_lstm_4,
                            activity_regularizer=activity_reg_lstm_4,
                            recurrent_regularizer=recurrent_reg_lstm_4,
                            dropout=hp.Float(
                                "lstm_4_dropout_rate", 
                                min_value=0.0, 
                                max_value=0.8, 
                                step=0.1
                            )
                        )
                    )
    print("Adding first Dense layer...")
    dense_1_units = hp.Int(
        "dense_input_unit",
        min_value=16, 
        max_value=512,
        step=16
    )
    kernel_reg_dense_1 = RegWrapper(
        hp.Choice("kernel_dense_1_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "kernel_dense_1_reg_value", 
            min_value=0, 
            max_value=0.5, 
            step=0.01
        )
    )
    bias_reg_dense_1 = RegWrapper(
        hp.Choice("bias_dense_1_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "bias_dense_1_reg_value", 
            min_value=0, 
            max_value=0.5, 
            step=0.01
        )
    )
    activity_reg_dense_1 = RegWrapper(
        hp.Choice("activity_dense_1_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "activity_dense_1_reg_value", 
            min_value=0, 
            max_value=0.5, 
            step=0.01
        )
    )
    model.add(
        Dense(
            dense_1_units, 
            activation=relu,
            kernel_regularizer=kernel_reg_dense_1,
            bias_regularizer=bias_reg_dense_1,
            activity_regularizer=activity_reg_dense_1
        )
    )
    
    print("Adding second Dense layer...")
    kernel_reg_dense_2 = RegWrapper(
        hp.Choice("kernel_dense_2_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "kernel_dense_2_reg_value", 
            min_value=0, 
            max_value=0.5, 
            step=0.01
        )
    )
    bias_reg_dense_2 = RegWrapper(
        hp.Choice("bias_dense_2_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "bias_dense_2_reg_value", min_value=0, max_value=0.5, step=0.01
        )
    )
    activity_reg_dense_2 = RegWrapper(
        hp.Choice("activity_dense_2_reg_type", ["l1", "l2", "l1l2"]), 
        hp.Float(
            "activity_dense_2_reg_value", 
            min_value=0, 
            max_value=0.5, 
            step=0.01
        )
    )
    model.add(
        Dense(
            1,
            kernel_regularizer=kernel_reg_dense_2,
            bias_regularizer=bias_reg_dense_2,
            activity_regularizer=activity_reg_dense_2
        )
    )
    print("Compiling model...")
    lr = hp.Float(
        "lr", 
        min_value=1e-4,
        max_value=1,
        step=1e-1
    )
    model.compile(
        loss="mse", 
        optimizer=Adam(learning_rate=lr), 
        metrics=[RootMeanSquaredError()]
    )
    return model

num_epochs=300
num_batch_size=24

tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=num_epochs,
    overwrite=True,
    hyperband_iterations=5
)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                              patience=5, min_lr=0.0001)

print("Fitting network...")

start = time.time()
tuner.search(
        x=X_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=num_batch_size,
        verbose=2,
        shuffle=False,
        validation_data=(X_test, y_test),
        callbacks = [reduce_lr]
)
end=time.time()

print(
    f"Time taken to search models: {str(datetime.timedelta(seconds=end-start))}"
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

hypermodel = tuner.hypermodel.build(best_hps)

hypermodel_history = hypermodel.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch_size, validation_data=(X_test, y_test), verbose=2, shuffle=False)

hypermodel.save("models/hypermodel_Jul28.h5")

pyplot.plot(hypermodel_history.history["loss"], label="train")
pyplot.plot(hypermodel_history.history["val_loss"], label="test")
pyplot.legend()
pyplot.savefig("models/val_loss_Jul28.png")
pyplot.show()

pdb.set_trace()






# fit network
#history = model.fit(X_train, y_train, epochs=100, batch_size=24, validation_data=(X_test, y_test), verbose=2, shuffle=False)
#print("Success!")

#def experiment(X_traein, X_test, y_train, y_test, )



