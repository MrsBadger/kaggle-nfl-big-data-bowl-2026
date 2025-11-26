import pandas as pd
import os
import glob
import re
import numpy as np

def create_training_rows(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        input_df.sort_values(["game_id","play_id","nfl_id","frame_id"])
                .groupby(["game_id","play_id","nfl_id"], as_index=False)
                .tail(1)
                .reset_index(drop=True)
                .rename(columns={"frame_id":"last_frame_id"})
    )

    out = output_df.copy()
    out = out.rename(columns={"x":"target_x","y":"target_y"})
    out["id"] = (
        out["game_id"].astype(str) + "_" +
        out["play_id"].astype(str) + "_" +
        out["nfl_id"].astype(str) + "_" +
        out["frame_id"].astype(str)
    )
    m = out.merge(agg, on=["game_id","play_id","nfl_id"], how="left", suffixes=("","_last"))
    m["delta_frames"] = (m["frame_id"] - m["last_frame_id"]).clip(lower=0).astype(float)
    m["delta_t"] = m["delta_frames"] / 10.0
    return m




class Data():
    def __init__(self, path=""):
        files_input = sorted(glob.glob(path + "train/input_2023_w*.csv"))
        files_output = sorted(glob.glob(path + "train/output_2023_w*.csv"))

        INPUT_DFS = []
        OUTPUT_DFS = []

        for f_in, f_out in zip(files_input, files_output):
            df_in = pd.read_csv(f_in)
            df_out = pd.read_csv(f_out)

            week_match = re.search(r"w(\d+)", f_out)
            week = int(week_match.group(1)) if week_match else None

            df_in["week"] = week
            df_out["week"] = week

            INPUT_DFS.append(df_in)
            OUTPUT_DFS.append(df_out)

            print(f"{f_in}: {df_in.shape}, {f_out}: {df_out.shape}, week: {week}")
        input_df = pd.concat(INPUT_DFS, ignore_index=True)
        output_df = pd.concat(OUTPUT_DFS, ignore_index=True)

        
        df = create_training_rows(input_df, output_df)

        self.data = df


    def preproc(self, type=""):
        X_LIMIT = 120
        Y_LIMIT = 53.3

        right_eda = self.data[self.data["play_direction"] == "right"].copy()
        left_eda = self.data[self.data["play_direction"] == "left"].copy()

        right_eda["was_left"] = 0
        left_eda["was_left"]  = 1

        left_eda["x"] = X_LIMIT - left_eda["x"]
        left_eda["y"] = Y_LIMIT - left_eda["y"]

        left_eda["ball_land_x"] = X_LIMIT - left_eda["ball_land_x"]
        left_eda["ball_land_y"] = Y_LIMIT - left_eda["ball_land_y"]

        # mirroring the future positions for consistency
        if "target_x" in left_eda.columns:
            left_eda["target_x"] = X_LIMIT - left_eda["target_x"]
        if "target_y" in left_eda.columns:
            left_eda["target_y"] = Y_LIMIT - left_eda["target_y"]

        left_eda["dir"] = (left_eda["dir"] + 180) % 360
        left_eda["o"] = (left_eda["o"] + 180) % 360

        left_eda["play_direction"] = "right"

        df = pd.concat([right_eda, left_eda], ignore_index=True)

        #targets = df[df["player_to_predict"] == True].copy()

        #df["target_dx"] = df["ball_land_x"] - df["x"]
        #df["target_dy"] = df["ball_land_y"] - df["y"]

        exclude_cols = [
            "player_name", "player_position", "player_role", "player_side", "play_direction",
            "player_height", "player_birth_date",
            "game_id", "play_id", "nfl_id",
            "x_out", "y_out", "frame_id_in", "frame_id_out",
            "target_dx", "target_dy",
        ]

        feature_cols = [c for c in self.data.columns if c not in exclude_cols]

        self.feature_cols = feature_cols

        self.preprocessed = df



class PredictionSequencer():
    def __init__(self, model, features = []):
        """Idea is to take a model here and then to allow it to be used consecutively,
        takes a model and the input features, other than that the class has the attributes
        predictions, containing all teh predictions (total n), and the inputs, corresponding
        to the submitted and then created input features."""
        self.model = model
        self.features = features
        self.predictions = []
        self.inputs = []
        self.datas = []

    def create_new_data(self, start_dpoint, prediction, data):
        """Here the predicted next point should be used to calculate
        the next input to the model. The significance being that the
        new X and Y values should correspond to a new observation."""
        process = "include preprocessing from maybe data class?"
        previous = data[(data["game_id"] == start_dpoint["game_id"]) & \
                        (data["play_id"] == start_dpoint["play_id"]) & \
                        (data["nfl_id"] == start_dpoint["nfl_id"])]
        previous_f = previous.sort_values("frame_id").iloc[-1]
        new_dpoint = previous_f.copy()

        columns_calculated = ["x", "y", "s", "a", "o", "dir"]
        x1 = prediction[0]
        y1 = prediction[1]

        x0 = previous_f['x']
        y0 = previous_f['y']

        dx = x1 - x0
        dy = y1 - y0
        t = 0.1
        # not sure about the orientation calcualtion here
        o = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        s = np.sqrt(dx**2 + dy**2) / t

        s0 = previous_f["s"]
        a = (s - s0) / t
        # direction just taken to be the same as the previous value
        #dir1 = np.degrees(np.arctan2(dy, dx))
        new_dpoint["x"] = x1
        new_dpoint["y"] = y1
        new_dpoint["s"] = s
        new_dpoint["a"] = a
        new_dpoint["o"] = o
        #new_dpoint["dir"] = dir1
        new_dpoint["frame_id"] = new_dpoint["frame_id"] + 1
            
        return new_dpoint
    
    def feature_engineering(self, processed_datapoint, data, type="a"):
        """Here goes just the feature engineering."""

        # Join previous datapoint to the data.
        group_cols = ['game_id', 'play_id', 'nfl_id']
        df = data.copy()
        df = pd.concat([df, pd.DataFrame([processed_datapoint])], ignore_index=True)
        

        if type == "a":
            group = df.groupby(group_cols)

            for lag in [1, 2]:#, 3, 5]:
                for c in ['x', 'y']:#, 'velocity_x', 'velocity_y', 's']:
                    df[f"{c}_lag{lag}"] = group[c].shift(lag)

            self.datas.append(df)
            outp = df.iloc[-1]
            return outp

        return 

    def predict(self, start_dpoint, data, n):
        """Taking a starting point start_dpoint, we make n predictions
        and using a some starting data and logic defined in process_prediciton output,
        we just call model.predict() in the familiar sklearn syntax to make the desred
        number of predictions."""
        self.predictions = []
        self.inputs = []
        for i in range(n):
            if i==0:
                prediction = self.model.predict(start_dpoint)
            else:
                prediction = self.model.predict(self.inputs[-1])
            new_datapoint = self.create_new_data(self, start_dpoint, prediction, data)
            
            processed = self.feature_engineering(new_datapoint, data)
            self.inputs.append(processed)
            self.predictions.append(prediction)
        return self.predictions, self.inputs

    