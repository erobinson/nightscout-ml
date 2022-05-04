from nightscout_ml_base import NightscoutMlBase
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
# import autosklearn.classification


class SimpleModel(NightscoutMlBase):
    
    def build_model(self):
        df = pd.read_csv(self.data_folder+'/simple_model_data.csv')
        # df = df.sample(frac=1).reset_index(drop=True)

        label_cols = ["sgv", "total_iob", "cob"]
        df_features = df[label_cols]
        # df_features = df.drop("aismb")
        # df_features_np = np.array(df_features)
        df_labels = df[["aismb"]]

        normalize = layers.Normalization()
        normalize.adapt(np.array(df_features))

        model = tf.keras.Sequential([
            normalize,
            layers.Dense(3),
            layers.Dense(1)
            ])

        model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

        model.fit(df_features, df_labels, epochs=8)

        print("\n\n== tf neural networkk ==")
        low = model.predict([50.0,0.0,0.0])
        low_w_iob = model.predict([50.0,3.0,0.0])
        normal_w_iob = model.predict([100.0,3.0,0.0])
        normal_wo_iob = model.predict([100.0,0.0,0.0])
        print(f"    low: {low}    low_w_iob: {low_w_iob}    normal_w_iob: {normal_w_iob}    normal_wo_iob: {normal_wo_iob}")
        high_sgv = model.predict([200.0,0.0,0.0])
        high_cob = model.predict([100.0,0.0,30.0])
        high_both = model.predict([200.0,0.0,30.0])
        print(f"    high_sgv: {high_sgv}    high_cob: {high_cob}    high_both: {high_both}")
        

    def sklearn_linear_regression_model(self):
        df = pd.read_csv(self.data_folder+'/simple_model_data.csv')
        # df = df.sample(frac=1).reset_index(drop=True)

        label_cols = ["sgv", "total_iob", "cob"]
        df_features = df[label_cols]
        # df_features = df.drop("aismb")
        # df_features_np = np.array(df_features)
        df_labels = df[["aismb"]]
        predictor = LinearRegression(n_jobs=-1)
        predictor.fit(X=df_features.values, y=df_labels.values)

        print("\n\n== sklearn Linear Regression ==")
        low = predictor.predict([[50.0,0.0,0.0]])
        low_w_iob = predictor.predict([[50.0,3.0,0.0]])
        normal_w_iob = predictor.predict([[100.0,3.0,0.0]])
        normal_wo_iob = predictor.predict([[100.0,0.0,0.0]])
        print(f"    low: {low}    low_w_iob: {low_w_iob}    normal_w_iob: {normal_w_iob}    normal_wo_iob: {normal_wo_iob}")
        high_sgv = predictor.predict([[200.0,0.0,0.0]])
        high_cob = predictor.predict([[100.0,3.0,30.0]])
        high_both = predictor.predict([[200.0,3.0,30.0]])
        print(f"    high_sgv: {high_sgv}    high_cob: {high_cob}    high_both: {high_both}")

    def build_tf_regression(self):
        df = pd.read_csv(self.data_folder+'/simple_model_data.csv')
        
        train_dataset = df.sample(frac=0.8, random_state=0)
        test_dataset = df.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop('aismb')
        test_labels = test_features.pop('aismb')

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        linear_model = tf.keras.Sequential([
            normalizer,
            layers.Dense(units=1)
        ])

        linear_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error')

        linear_model.fit(
            train_features,
            train_labels,
            epochs=10,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split = 0.2)

        test_results = {}
        test_results['linear_model'] = linear_model.evaluate(
            test_features, test_labels, verbose=0)

        now = datetime.now()
        date_str = "{}-{}-{}_{}:{}".format(now.year, now.month, now.day, now.hour, now.minute)
        linear_model.save('models/tf_linear_model_'+date_str)

        print(test_results)





    # Test with AutoSKLearn but wouldn't install
    # def build_auto_sklearn(self):
    #     df = pd.read_csv(self.data_folder+'/simple_model_data.csv')
    #     label_cols = ["sgv", "total_iob", "cob"]
    #     X_train = df[label_cols]
    #     # df_features = df.drop("aismb")
    #     # df_features_np = np.array(df_features)
    #     y_train = df[["aismb"]]
    #     cls = autosklearn.classification.AutoSklearnClassifier()
    #     cls.fit(X_train, y_train)
    #     predictions = cls.predict([[50.0,0.0,0.0]])
