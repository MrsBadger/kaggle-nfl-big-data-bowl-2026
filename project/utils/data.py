import pandas as pd
import os
import glob
import re

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
        self.model = model
        self.features = features
        self.predictions = []
        self.inputs = []

    def process_prediction_output(self, prediction, data):
        process = "include preprocessing from maybe data class?"
        return "processed new row for prediciton in next step"

    def predict(self, start_dpoint, n):
        self.predictions = []
        self.inputs = []
        for i in range(n):
            if n==0:
                prediction = self.model.predict(start_dpoint)
            else:
                prediction = self.model.predict(self.inputs[-1])
            processed = self.process_prediction_output(prediction, prediction, start_dpoint)
            self.inputs.append(processed)
        return self.predictions, self.inputs

    