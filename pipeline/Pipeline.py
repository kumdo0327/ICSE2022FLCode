import os
import time
from read_data.ManyBugsDataLoader import ManyBugsDataLoader
from read_data.Defects4JDataLoader import Defects4JDataLoader
from read_data.SIRDataLoader import SIRDataLoader
from data_process.data_systhesis.resampling import ResamplingData
from data_process.data_systhesis.smote import SMOTEData
from data_process.data_systhesis.cvae_synthesis import CVAESynthesisData
from data_process.dimensional_reduciton.PCA import PCAData
from data_process.data_undersampling.undersampling import UndersamplingData
from calculate_suspiciousness.CalculateSuspiciousness import CalculateSuspiciousness

class Pipeline:
    def __init__(self, project_dir, configs):
        self.configs = configs
        self.project_dir = project_dir
        self.dataset = configs["-d"]
        self.program = configs["-p"]
        self.bug_id = configs["-i"]
        self.experiment = configs["-e"]
        self.method = configs["-m"].split(",")
        self.dataloader = self._choose_dataloader_obj()
        
        self.start = time.time()

    def run(self):
        self._run_task()
        time_log = int(time.time() - self.start) 

    def _dynamic_choose(self, loader):
        self.dataset_dir = os.path.join(self.project_dir, "data")
        data_obj = loader(self.dataset_dir, self.program, self.bug_id)
        data_obj.load()
        return data_obj

    def _choose_dataloader_obj(self):
        if self.dataset == "d4j":
            return self._dynamic_choose(Defects4JDataLoader)
        if self.dataset == "manybugs" or self.dataset == "motivation":
            return self._dynamic_choose(ManyBugsDataLoader)
        if self.dataset == "SIR":
            return self._dynamic_choose(SIRDataLoader)

    def _run_task(self):
        if self.experiment == "origin":
            self.data_obj = self.dataloader
        elif self.experiment == "resampling":
            self.data_obj = ResamplingData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "undersampling":
            self.data_obj = UndersamplingData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "smote":
            self.data_obj = SMOTEData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "cvae":
            self.data_obj = CVAESynthesisData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "fs":
            cp = float(self.configs["-cp"])
            ep = float(self.configs["-ep"])
            self.data_obj = PCAData(self.dataloader, "/volume/aeneas/cache", os.path.join(self.project_dir, "time/io"))
            self.data_obj.process(cp, ep)
            if self.bug_id == 'cache':
                return
            
        elif self.experiment == "fs_cvae":
            cp = float(self.configs["-cp"])
            ep = float(self.configs["-ep"])
            print('Pipeline.py : self.data_obj = PCAData(self.dataloader, "/volume/aeneas/cache", os.path.join(self.project_dir, "time/io"))')
            self.data_obj = PCAData(self.dataloader, "/volume/aeneas/cache", os.path.join(self.project_dir, "time/io"))
            print('Pipeline.py : self.data_obj.process(cp, ep)')
            self.data_obj.process(cp, ep)
            print('Pipeline.py : self.data_obj = CVAESynthesisData(self.data_obj)')
            self.data_obj = CVAESynthesisData(self.data_obj)
            print('Pipeline.py : self.data_obj.process()')
            self.data_obj.process()

        print('CalculateSuspiciousness')
        for m in self.method:
            save_rank_path = os.path.join(self.project_dir, f"results/{m}/{self.program}")
            if not os.path.exists(save_rank_path):
                os.makedirs(save_rank_path)
            cc = CalculateSuspiciousness(self.data_obj, self.method, save_rank_path, self.experiment, int(time.time() - self.start))
            time_log = cc.run()

            dir = os.path.join(self.project_dir, f"time/e2e/{m}/{self.program}")
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(os.path.join(dir, f"{self.bug_id}.txt", "w")) as f:
                f.write(f"{time_log // 3600}:{(time_log % 3600) // 60}:{time_log % 60}")
