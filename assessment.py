"""





"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
import datetime

from matplotlib.dates import  DateFormatter

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

import shap

from xgboost import XGBRegressor

################################################################################
# %% DEF MODEL CREATION CLASS
################################################################################

class SteamProductionModelTesting():

    def __init__(self, fname, column_dict=None, target='steam', smooth_signals=[], window=5):

        self.fname = fname
        self.column_dict = column_dict
        self.load_data()
        self.filter_data()
        if smooth_signals:
            self.smooth_signal(smooth_signals, window)
        self.X = self.data.copy()
        self.X.drop([target, 'datetime'], axis=1, inplace=True)
        self.y = self.data[target].copy()


    def load_data(self):
        """
        Load raw data, and replace column names if applicable
        """
        ##### READ DATA
        self.raw_data = pd.read_csv(fname, parse_dates=['utctimestamp'])

        ##### REPLACE COLUMN NAMES
        if self.column_dict:
            self.raw_data.columns = [self.column_dict[signal] for signal in self.raw_data.columns]


    def filter_data(self):
        """
        Filter data according to rpm, ammonia consumption and steam production
        """
        ##### FILTER OUT DOWNTIME
        self.data = self.raw_data[self.raw_data['rpm']>4900]

        ##### FILTER NEGATIVE AMMONIA CONSUMPTION
        self.data = self.data[self.data['ammonia'] > 200]

        ##### FILTER LOW STEAM PRODUCTION
        self.data = self.data[self.data['steam'] > 0]

        ##### FIND MISSING VALUES PER PARAMETER
        for parameter in self.data.columns:
            print(f"{parameter} has {self.raw_data[parameter].isna().sum()} missing values")

        ##### DROP ROWS WITH MISSING DATA
        self.data = self.data.dropna()


    def smooth_signal(self, signals, window=5):
        """
        Smooth signal by rolling mean
        """

        ##### ATTEPT TO DENOISE MEASUREMENT
        for signal in signals:
            print(f'smoothing {signal}')
            self.data[signal] = self.data[signal].rolling(window).mean()

        self.data = self.data.iloc[range(23, len(self.data))]


    def run_pipeline(self, estimator):
        """
        Build a Sklean type pipeline for multiple model testing
        """

        self.pipeline =  Pipeline([('scaler', StandardScaler()), ('estimator', estimator)])

        cv_score = cross_val_score(self.pipeline, self.X, self.y, cv=10)

        print(f"Model {self.pipeline['estimator']} CV scores: {cv_score}")
        print(f"Avg. CV scores {np.mean(cv_score)}")


################################################################################
# %% MAIN CALL
################################################################################

if __name__ == '__main__':

    fname = "NA7_20201007_1h_task.csv"

    column_dict = {
        'utctimestamp' : 'datetime',
        '01FI1103/AI1/PV.CV' : 'steam',
        '01FI1101E/PV.CV' : 'ammonia',
        '01TI1538/AI1/PV.CV' : 'T_amb',
        '60PI0496/AI1/PV.CV' : 'p_amb',
        '01AI1923/AI1/PV.CV' : 'h_amb',
        '01HC1955/PID1/PV.CV' : 'rpm'
        }

    model_dev = SteamProductionModelTesting(fname, column_dict, target='steam', smooth_signals=['steam'], window=10)

    ################################################################################
    # %% PLOT TIME SERIES DATA
    ################################################################################

    ##### PLOT FULL DATA SET
    fig, axs = mp.subplots(model_dev.raw_data.shape[1]-1, 1, figsize=(12,15))
    for i, ax in enumerate(axs):
        model_dev.raw_data.plot(x='datetime', y=model_dev.raw_data.columns[i+1], ax=ax)
        ax.set_title(model_dev.raw_data.columns[i+1])
        #ax.xaxis.set_major_formatter(DateFormatter('%Y-%m') )
    fig.tight_layout()
    mp.savefig('figures/raw_data_time_plot.png')

    ##### PLOT FIRST DOWNTIME EVENT
    fig, axs = mp.subplots(model_dev.raw_data.shape[1]-1, 1, figsize=(12,15))
    for i, ax in enumerate(axs):
        model_dev.raw_data.plot(x='datetime', y=model_dev.raw_data.columns[i+1], ax=ax)
        ax.set_title(model_dev.raw_data.columns[i+1])
        #ax.xaxis.set_major_formatter(DateFormatter('%Y-%m') )
        ax.set_xlim([datetime.date(2017, 4, 1), datetime.date(2017, 4, 10)])
    fig.tight_layout()
    mp.savefig('figures/raw_data_downtime.png')

    ################################################################################
    # %% PLOT PAIRPLOT
    ################################################################################

    ##### PLOT BASIC STATISTICS BASED ON FILTERED DATA
    sb.pairplot(model_dev.data)
    mp.savefig('figures/filtered_pairplot.png')

    ################################################################################
    # %% RUN CROSSVALIDATION ON SUPPLIED METHODS
    ################################################################################

    methods = [
        LinearRegression(),
        ElasticNet(alpha=0.01, l1_ratio=0.01),
        XGBRegressor(),
        RandomForestRegressor(max_depth=10, n_estimators=20),
        ]

    for method in methods:
        model_dev.run_pipeline(method)

    ################################################################################
    # %% RUN CROSSVALIDATION ON SUPPLIED METHODS
    ################################################################################

    rf = RandomForestRegressor(max_depth=10, n_estimators=20)
    rf.fit(model_dev.X, model_dev.y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(model_dev.X)
    shap.summary_plot(shap_values, model_dev.X, show=False)
    mp.savefig('figures/rf_shap_value.png')

################################################################################
# %% BACKUP
################################################################################

################################################################################
# %% VIF CHECK
################################################################################

pd.Series([variance_inflation_factor(X_train_norm.values, i)
               for i in range(X_train_norm.shape[1])],
              index=X_train_norm.columns)

################################################################################
# %% NEURAL NETWORK TESTING
################################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()


model.add(Dense(10, input_shape=(5,), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')
model.summary()

history = model.fit(
    X_train_norm,
    y_train_norm,
    epochs=100,
    validation_data=(X_test_norm, y_test_norm)
)

# %%

y_pred = model.predict(X_train_norm)
score = r2_score(y_train_norm, y_pred)
print(f'NN model train score: {score}')

y_pred = model.predict(X_test_norm)
score = r2_score(y_test_norm, y_pred)
print(f'NN model test score: {score}')

# %%

mp.plot(history.history['loss'])
mp.plot(history.history['val_loss'])
