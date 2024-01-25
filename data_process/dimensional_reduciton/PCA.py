import math
import os
import time

import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData

class PCAData(ProcessedData):

    def __init__(self, raw_data, cache_path, time_path):
        super().__init__(raw_data)
        self.rest_columns = None

        self.feature_path = os.path.join(cache_path, self.program) + "-feature.npy"
        self.time_path = time_path

    def process(self, components_percent=0.7, eigenvalue_percent=0.7):
        if len(self.label_df) > 1 or True: ## 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
            covMatrix = self.feature_df.cov()

            featValue, featVec = None, None
            if os.path.exists(self.feature_path):
                print('PCA.py : load cache')
                begin = time.time()
                with open(self.feature_path, 'rb') as f:
                    featValue = np.load(f)
                    featVec = np.load(f)
                end = int(time.time() - begin)
                print('\tdone')
                with open(os.path.join(self.time_path,  f"read/{self.program}-{self.bug_id}.txt"), "w") as f:
                    f.write(f"{end // 3600}:{(end % 3600) // 60}:{end % 60}")
            else:
                print('PCA.py : featValue, featVec = np.linalg.eig(covMatrix)')
                begin = time.time()
                featValue, featVec = np.linalg.eig(covMatrix)
                end = int(time.time() - begin)
                print('\tdone')
                with open(os.path.join(self.time_path,  f"eig/{self.program}.txt"), "w") as f:
                    f.write(f"{end // 3600}:{(end % 3600) // 60}:{end % 60}")

                print('PCA.py : caching')
                with open(self.feature_path, 'wb') as f:
                    np.save(f, featValue)
                    np.save(f, featVec)
                print('\tdone')
                return


            print('PCA.py : trunc by ep')
            index = np.argsort(-featValue)
            eigenvalue_num = math.trunc(len(self.feature_df.values[0]) * eigenvalue_percent)
            selected_values = featValue[index[:eigenvalue_num]]
            selected_vectors = featVec.T[index[:eigenvalue_num]].T
            print('\tdone')

            print('PCA.py : init contri')
            contri = np.array([sum(v) for v in np.abs(selected_vectors)])
            contri_index = np.argsort(-contri)
            print('\tdone')

            print('PCA.py : trunc by cp')
            num_components = math.trunc(len(self.feature_df.values[0]) * components_percent)
            selected_index = contri_index[:num_components]
            rest_index = contri_index[num_components:]
            rest_columns = self.feature_df.columns[rest_index]
            self.rest_columns = list(rest_columns)
            low_features = self.feature_df.values.T[selected_index].T
            print('\tdone')

            print('PCA.py : set self.feature_df, self.label_df, self.data_df')
            columns = self.feature_df.columns[selected_index]
            low_features = pd.DataFrame(low_features, columns=columns)
            low_data = pd.concat([low_features, self.label_df], axis=1)

            self.feature_df = low_features
            self.label_df = self.label_df
            self.data_df = low_data
            print('\tdone')
