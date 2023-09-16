import pandas as pd
import os
from datasets import Dataset


class EmotionDataLoader:
    def __init__(self):
        self.data_path = "data"
        self.id2label = {0: "anger",
                         1: "fear",
                         2: "disgust",
                         3: "joy",
                         4: "sadness",
                         5: "surprise"}

        self.label2id = {value: key for key, value in self.id2label.items()}


    def load_GoEM(self, train_file, dev_file, test_file):
        folder = "GoEM"
        train_file_path = os.path.join(self.data_path, folder, train_file)
        dev_file_path = os.path.join(self.data_path, folder, dev_file)
        test_file_path = os.path.join(self.data_path, folder, test_file)

        # Loading .tsv files from Go Emotion dataset
        train_df = pd.read_table(train_file_path)
        dev_df = pd.read_table(dev_file_path)
        test_df = pd.read_table(test_file_path)

        # To avoid repetetive operations
        df_list = [train_df, dev_df, test_df]

        # Creating a mapping between all 28 emotions and their respective int
        int2str = {}
        with open(os.path.join(self.data_path, folder, "emotions.txt"), encoding="utf8") as f:
            for i, emotion in enumerate(f):
                int2str[i] = emotion.strip()

        # Dropping 3rd column & removing examples with num. of labels > 1
        for i in range(3):
            df_list[i] = df_list[i].drop(df_list[i].columns[2], axis=1)
            df_list[i].columns = ["text", "labels"]

            # Dropping multilabel samples
            df_list[i]["multilabel"] = df_list[i].labels.str.split(",").apply(lambda x: len(x) > 1)
            df_list[i] = df_list[i][df_list[i]["multilabel"] == False]
            df_list[i] = df_list[i].drop(["multilabel"], axis=1)

        # Mapping all 28 emotions to Ekman's 6 basic emotions
        for i in range(3):
            df_list[i]["labels"] = df_list[i]["labels"].apply(lambda x: int2str[int(x)]).apply(lambda x: self.ekman_mapping(x))
            df_list[i] = df_list[i].mask(df_list[i].eq('None')).dropna()
        
        return df_list


    def load_EDFER(self, train_file, dev_file, test_file):
        # Very similar function to the other load_GoEM, just sligtly simpler
        folder = "EDFER"
        train_file_path = os.path.join(self.data_path, folder, train_file)
        dev_file_path = os.path.join(self.data_path, folder, dev_file)
        test_file_path = os.path.join(self.data_path, folder, test_file)

        train_df = pd.read_csv(train_file_path)
        dev_df = pd.read_csv(dev_file_path)
        test_df = pd.read_csv(test_file_path)

        df_list = [train_df, dev_df, test_df]

        int2str = {0: "sadness",
                   1: "joy",
                   2: "love",
                   3: "anger",
                   4: "fear",
                   5: "surprise"}

        for i in range(3):
            df_list[i].rename(columns={"label": "labels"}, inplace=True)
            df_list[i]["labels"] = df_list[i]["labels"].apply(lambda x: int2str[int(x)]).apply(lambda x: self.ekman_mapping(x))
            
        return df_list


    def combine_dataframes(self, first_df_list, second_df_list):
        combined_dfs = []
        random_state = 77 #  random state for pandas sampling
        for i in range(3):
            combined_dfs.append(pd.concat([first_df_list[i], second_df_list[i]], axis=0))
            combined_dfs[i]["labels"] = combined_dfs[i]["labels"].apply(lambda x: self.label2id[x])
            combined_dfs[i] = combined_dfs[i].sample(frac=0.7, replace=False, random_state=random_state)
            combined_dfs[i] = Dataset.from_pandas(combined_dfs[i])
        
        return combined_dfs

    
    def get_id2label(self):
        return self.id2label

    def get_label2id(self):
        return self.label2id

            
    def ekman_mapping(self, label):
        if label in ("anger", "annoyance", "disapproval"):
            return "anger"

        elif label in ("disgust"):
            return "disgust"

        elif label in ("fear", "nervousness"):
            return "fear"

        elif label in ("joy", "amusement", "approval", "excitement", "gratitude", "love",
                       "optimism", "relief", "pride", "admiration", "desire", "caring"):
            return "joy"

        elif label in ("sadness", "disappointment", "embarrassment", "grief", "remorse"):
            return "sadness"

        elif label in ("surprise", "realization", "confusion", "curiosity"):
            return "surprise"
        