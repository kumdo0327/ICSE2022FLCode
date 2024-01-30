import os
import numpy as np
from metrics.calc_corr import calc_corr
from utils.write_util import write_corr_to_txt, write_rank_to_txt
from utils.read_util import get_corr


class CalculateSuspiciousness():
    def __init__(self, data_obj, method, save_rank_path, experiment, time):
        self.data_obj = data_obj
        self.method = method
        self.sava_rank_path = save_rank_path
        self.suspicious_list = None
        self.state = experiment
        self.time = time
        self.start = time.time()

    def run(self):
        self._calculate_susp_for_method_list()
        self._calculate_rank()
        self._save_rank()
        self._calculate_AR_rank()
        self._save_AR_rank()
        return self.time + int(time.time() - self.start)

    def _calculate_susp_for_method_list(self):
        for method in self.method:
            self._calculate_susp_for_method(method)

    def _calculate_susp_for_method(self, method):
        self.suspicious_list = calc_corr(self.data_obj.data_df, method)
        for col in self.data_obj.rest_columns:
            self.suspicious_list[col] = 0
        write_corr_to_txt(method, self.suspicious_list, self.data_obj.file_dir, self.state, self.data_obj.bug_id)

    def _calculate_rank(self):
        all_df_dict = get_corr(self.data_obj.file_dir, self.method, self.state, self.data_obj.bug_id)
        self.rank_MFR_dict = self.__calculate_rank(all_df_dict, self.data_obj.fault_line, self.method)

    def _calculate_AR_rank(self):
        all_df_dict = get_corr(self.data_obj.file_dir, self.method, self.state, self.data_obj.bug_id)
        self.rank_MAR_dict = self.__calc_MAR_rank(all_df_dict, self.data_obj.fault_line, self.method)

    def _save_rank(self):
        save_rank_filename = os.path.join(self.sava_rank_path, f"{self.state}/{self.data_obj.program}/{self.data_obj.bug_id}-FR.txt")
        write_rank_to_txt(self.rank_MFR_dict, save_rank_filename)

    def _save_AR_rank(self):
        save_rank_filename = os.path.join(self.sava_rank_path, f"{self.state}/{self.data_obj.program}-{self.data_obj.bug_id}-AR.txt")
        write_rank_to_txt(self.rank_MAR_dict, save_rank_filename)

    def __calculate_rank(self, all_df_dict, fault_line_data, method_list):
        real_fault_line_data = list()

        real_line_data = all_df_dict[method_list[0]]['line_num'].tolist()
        for line in fault_line_data:
            if line in real_line_data:
                real_fault_line_data.append(line)

        result_dict = dict()
        for method in method_list:
            result_dict[method] = float('-inf')
        for method in method_list:
            result_dict[method] = min(self.rank(method, all_df_dict[method], real_fault_line_data))
        return result_dict

    def __calc_MAR_rank(self, all_df_dict, fault_line_data, method_list):
        real_fault_line_data = list()

        real_line_data = all_df_dict[method_list[0]]['line_num'].tolist()
        for line in fault_line_data:
            if line in real_line_data:
                real_fault_line_data.append(line)

        result_dict = dict()
        for method in method_list:
            result_dict[method] = float('-inf')
        for method in method_list:
            result_dict[method] = np.mean(self.rank(method, all_df_dict[method], real_fault_line_data))
        return result_dict

    def rank(self, method, df, fault_line_data) -> list:
        ranking = [0 for _ in range(len(df.index) + 1)]
        virtual_ranking = 0
        lowest = df[method][0]
        cand = list()

        for i in df.index:
            if lowest > df[method][i]:
                lowest = df[method][i]
                for line in cand:
                    ranking[line] = virtual_ranking
                cand.clear()
            cand.append(df["line_num"][i])
            virtual_ranking += 1

        for line in cand:
            ranking[line] = len(df.index)
            
        faults_ranking = [ranking[line] for line in fault_line_data]
        return faults_ranking
