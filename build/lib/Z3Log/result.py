import os
import re
import natsort

from typing import List
from copy import deepcopy

from .stats import Stats
from .specs import Specs
from .config.path import *
from .utils import *
import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np

class Result:
    def __init__(self, specifications: Specs):
        self.__benchmark: str = get_pure_name(specifications.benchmark)
        self.__approximate_benchmark: str = get_pure_name(specifications.approximate_benchmark)
        self.__experiment: str = specifications.experiment
        self.__strategy: str = specifications.strategy
        self.__optimization: str = specifications.optimization
        self.__metric: str = specifications.metric
        self.__precision: int = specifications.precision



        # Todo: Make this part a bit more generic!
        # Sometimes we want the result of only one benchmark
        # Sometimes we want the result of a family of benchmarks or a whole experiment
        self.__in_paths: List[str] = self.find_input_report_paths()
        self.__reports: List[Stats] = self.import_reports()


    @property
    def benchmark(self):
        return self.__benchmark

    @property
    def approximate_benchmark(self):
        return self.__approximate_benchmark

    @property
    def experiment(self):
        return self.__experiment

    @property
    def strategy(self):
        return self.__strategy

    @property
    def optimization(self):
        return self.__optimization

    @property
    def metric(self):
        return self.__metric

    @property
    def precision(self):
        return self.__precision

    @property
    def reports(self):
        return self.__reports

    @reports.setter
    def reports(self, reports: List[Stats]):
        self.__reports = deepcopy(reports)

    @property
    def in_paths(self):
        return self.__in_paths

    @in_paths.setter
    def in_paths(self, list):
        self.__in_paths = list

    # TODO
    # fix naming: include num_samples

    def find_relevant_csv(self):
        relevant_csv = []
        folder, extension = INPUT_PATH['report']

        if self.metric == WRE:
            if (self.optimization == OPTIMIZE or self.optimization == MAXIMIZE) and (self.strategy != BISECTION):
                cur_dir = f'{folder}/{self.benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}'
            else:
                cur_dir = f'{folder}/{self.benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}'
        else:
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                cur_dir = f'{folder}/{self.benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}'
            else:
                cur_dir = f'{folder}/{self.benchmark}_{self.experiment}_{self.metric}_{self.strategy}'


        print(f'{cur_dir = }')
        all_csv = [f for f in os.listdir(cur_dir)]

        print(f'{all_csv = }')
        for f in all_csv:
            if f.endswith(extension):
                relevant_csv.append(f'{cur_dir}/{f}')

        relevant_csv = natsort.natsorted(relevant_csv)  # sorting the files alphabetically in descending order

        return relevant_csv

    def find_input_report_paths(self):
        input_paths = []
        folder, extension = INPUT_PATH['report']
        if self.experiment == SINGLE:
            input_paths = self.find_relevant_csv()
            print(f'{input_paths = }')
        elif self.experiment == QOR:
            if self.metric == WRE:
                if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                    input_paths.append(
                        f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}.{extension}')
                else:
                    input_paths.append(
                        f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}.{extension}')
            else:
                if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                    input_paths.append(
                        f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}.{extension}')
                else:
                    input_paths.append(
                        f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}.{extension}')

        elif self.experiment == RANDOM:
            pass

        return input_paths

    def import_reports(self):
        reports: List = []
        for report_path in self.in_paths:
            reports.append(Stats(report_path))
        return reports

    def find_relevant_paths(self, folder: str, extension: str):

        all = [f for f in os.listdir(folder)]
        labelling_folders = []
        qor_paths = []

        for f in all:
            if re.search(self.metric, f):
                if f.endswith(extension):
                    qor_paths.append(f'{folder}/{f}')
                else:
                    labelling_folders.append(f'{folder}/{f}')

        labelling_paths = []
        for fold in labelling_folders:
            cur_files = [f for f in os.listdir(fold)]
            for c_file in cur_files:
                if c_file.endswith(extension):
                    labelling_paths.append(f'{fold}/{c_file}')

        relevant_paths = labelling_paths + qor_paths
        self.in_paths = []
        self.in_paths = relevant_paths

    def extract_fields_for_three_strategies(self):

        monotonic_files = []
        kind_files = []
        bisection_files = []

        for report_path in self.in_paths:
            if re.search(MONOTONIC, report_path):
                monotonic_files.append(report_path)
            elif re.search(KIND_BISECTION, report_path):
                kind_files.append(report_path)
            elif re.search(BISECTION, report_path):
                bisection_files.append(report_path)



        monotonic_files = natsort.natsorted(monotonic_files)
        kind_files = natsort.natsorted(kind_files)
        bisection_files = natsort.natsorted(bisection_files)

        count = 0
        for i in range(len(monotonic_files)):
            mono = monotonic_files[i].replace(MONOTONIC, '')
            kind = kind_files[i].replace(KIND_BISECTION, '')
            bisec = bisection_files[i].replace(BISECTION, '')
            if mono == kind == bisec:
                count += 1

        if count == len(monotonic_files):
            print(f'TEST -> PASS')


        monotonic_reports = []
        kind_reports = []
        bisection_reports = []

        for report_path in monotonic_files:
            monotonic_reports.append(Stats(report_path))
        for report_path in kind_files:
            kind_reports.append(Stats(report_path))
        for report_path in bisection_files:
            bisection_reports.append(Stats(report_path))

        return monotonic_reports, kind_reports, bisection_reports


    def collect_reports_for_three_strategies(self):
        folder, extension = OUTPUT_PATH['report']
        self.find_relevant_paths(folder, extension)
        monotonic_reports, kind_reports, bisection_reports = self.extract_fields_for_three_strategies()
        return monotonic_reports, kind_reports, bisection_reports

    def extract_fields_as_list(self, reports: List[Stats], field):
        field_list = []
        i = 0
        for stat in reports:

            if field == TOTAL_RUNTIME:
                field_list.append(float(stat.total_runtime))
            elif field == N_SATS or field == N_UNSATS:
                field_list.append(int(stat.num_sats) + int(stat.num_unsats))
            else:
                print(f'ERROR!!! No such attribute in class Stats')
        return field_list

    def draw_scatter_plot_all(self, field: str):

        monotonic_reports, kind_reports, bisection_reports = self.collect_reports_for_three_strategies()


        folder, extension = OUTPUT_PATH['figure']
        os.makedirs(f'{folder}', exist_ok=True)
        filename1 = f'{folder}/{SCATTER}_{self.metric}_{MONOTONIC}_{KIND_BISECTION}.{extension}'
        filename2 = f'{folder}/{SCATTER}_{self.metric}_{MONOTONIC}_{BISECTION}.{extension}'

        # print(f'{monotonic_reports = }')
        monotonic_list = self.extract_fields_as_list(monotonic_reports, field)
        kind_list = self.extract_fields_as_list(kind_reports, field)
        bisection_list = self.extract_fields_as_list(bisection_reports, field)

        max_monotonic = max(monotonic_list)
        max_kind = max(kind_list)
        max_bisection = max(bisection_list)
        max_value1 = max(max_monotonic, max_kind)
        max_value2 = max(max_monotonic, max_bisection)

        if self.metric == WHD:
            max_value1 = max(max_value1, 100)
            max_value2 = max(max_value2, 100)


        if self.metric == WAE:
            color = 'blue'
        elif self.metric == WRE:
            color = 'red'
        elif self.metric == WHD:
            color = 'black'
        a = [1] * 10

        # monotonic_list = [1] * len(monotonic_list)

        plt.rc('font', size=SMALL_SIZE+6)  # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE+6)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE+5)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE+4)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE+4)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # plt.tight_layout()
        plt.rcParams["figure.figsize"] = (6, 5)
        # plt.tight_layout()


        fig1, ax1 = plt.subplots()
        ax1.locator_params(tight=True, nbins=4)

        ErrorEval = 'ErrorEval'
        # ax1.set( xlabel=f'bisection+ runtime', ylabel=f'{ErrorEval} runtime')
        ax1.set_xlabel(f'bisection+ runtime')
        ax1.set_ylabel(ylabel=f'{ErrorEval} runtime', labelpad=20)
        ax1.set_title(f'{self.metric.upper()}: {ErrorEval} vs. bisection+')
        plt.scatter(kind_list, monotonic_list, c=color, s=10, marker='o', edgecolors=color, linewidths=1, alpha=0.1)
        plt.plot(list(np.arange(0, (max_value1), 0.01)), list(np.arange(0, (max_value1), 0.01)), c='grey')


        plt.xscale('log')
        plt.yscale('log')
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
        if self.metric == WAE or self.metric == WRE:
            xticks = [10 ** el for el in [-2, -1, 0, 1, 2]]
            xlabels = xticks
            yticks = [10 ** el for el in [-2, -1, 0, 1, 2]]
            ylabels = yticks
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xlabels)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(ylabels)
        elif self.metric == WHD:
            xticks = [10 ** el for el in [-2, -1, 0, 1, 2]]
            xlabels = xticks
            yticks = [10 ** el for el in [-2, -1, 0, 1, 2]]
            ylabels = yticks
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xlabels)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(ylabels)
        ax1.yaxis.set_label_coords(-.1, .5)
        plt.savefig(f'{filename1}', bbox_inches='tight')
        plt.close()

        # Figure2: Monotonic vs bisection
        fig2, ax2 = plt.subplots()



        ax2.set_xlabel(f'{BISECTION} runtime')
        ax2.set_ylabel(ylabel=f'{ErrorEval} runtime', labelpad=15)
        ax2.set_title(f'{self.metric.upper()}: {ErrorEval} vs. {BISECTION}')
        plt.scatter(bisection_list, monotonic_list, c=color, s=10, marker='o', edgecolors=color, linewidths=0.5,  alpha=0.1)
        plt.plot(list(np.arange(0, (max_value2), 0.01)), list(np.arange(0, (max_value2), 0.01)), c='grey')

        plt.yscale('log')
        plt.xscale('log')

        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
        ax2.get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatter())

        if self.metric == WAE or self.metric == WRE:
            xticks = [10 ** el for el in [-2, -1, 0, 1, 2]]
            xlabels = xticks
            yticks = [10 ** el for el in [-2, -1, 0, 1, 2]]
            ylabels = yticks
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xlabels)
            ax2.set_yticks(yticks)
            ax2.set_yticklabels(ylabels)
        else:
            xticks = [10 ** el for el in [-2, -1, 0, 1, 2]]
            xlabels = xticks
            yticks = [10 ** el for el in [-2, -1, 0, 1, 2]]
            ylabels = yticks
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xlabels)
            ax2.set_yticks(yticks)
            ax2.set_yticklabels(ylabels)
        ax2.yaxis.set_label_coords(-.1, .5)
        plt.savefig(f'{filename2}', bbox_inches='tight')
        plt.close()

        pass

    def avg_runtime(self, field):
        monotonic_reports, kind_reports, bisection_reports = self.collect_reports_for_three_strategies()
        monotonic_list = self.extract_fields_as_list(monotonic_reports, field)
        kind_list = self.extract_fields_as_list(kind_reports, field)
        bisection_list = self.extract_fields_as_list(bisection_reports, field)




        total_count = len(monotonic_list)
        monotonic_is_less_than_kind = 0
        monotonic_is_less_than_bisection = 0


        less_than_100ms = 0


        bis_morethan_100s = 0
        kind_morethan_100s = 0

        savings_bis = 0
        savings_kind = 0
        for i in range(len(monotonic_list)):
            if monotonic_list[i] < kind_list[i]:
                monotonic_is_less_than_kind += 1
            if monotonic_list[i] < bisection_list[i]:
                monotonic_is_less_than_bisection += 1
            if self.metric == WHD:
                if monotonic_list[i] <= 0.1 and bisection_list[i] <= 0.1 and kind_list[i] <= 0.1:
                    less_than_100ms += 1

            savings_bis += monotonic_list[i] - bisection_list[i]
            savings_kind += monotonic_list[i] - kind_list[i]

            # if self.metric == WRE:
            #     if bisection_list[i] >= 100:
            #         bis_morethan_100s += 1
            #
            #     if kind_list[i] >= 100:
            #         kind_morethan_100s += 1



        print(f'monotonic < bisection+ (%) = {(monotonic_is_less_than_kind/total_count) * 100}%')
        print(f'monotonic < bisection (%) = {(monotonic_is_less_than_bisection / total_count) * 100}%')

        print(f'{self.metric = }')
        print(f'savings bisection = {savings_bis = }')
        print(f'savings bisection+ = {savings_kind = }')
        # print(f'kind savings in seconds = {monotonic_is_less_than_kind = }')
        # print(f'bisection savings in seconds = {monotonic_is_less_than_bisection = }')
        # print(f'{less_than_100ms/total_count = }')
        # print(f'{self.metric = }')
        # print(f'{bis_morethan_100s/total_count = }')
        # print(f'{kind_morethan_100s/total_count = }')


        # print(f'{len(monotonic_list) = }')


    def draw_avg_plot_all(self, field):
        monotonic_reports, kind_reports, bisection_reports = self.collect_reports_for_three_strategies()
        folder, extension = OUTPUT_PATH['figure']
        os.makedirs(f'{folder}', exist_ok=True)
        filename1 = f'{folder}/{AVERAGE}_{self.metric}_{MONOTONIC}_{KIND_BISECTION}.{extension}'
        filename2 = f'{folder}/{AVERAGE}_{self.metric}_{MONOTONIC}_{BISECTION}.{extension}'


        monotonic_list = self.extract_fields_as_list(monotonic_reports, field)
        kind_list = self.extract_fields_as_list(kind_reports, field)
        bisection_list = self.extract_fields_as_list(bisection_reports, field)
        # print(monotonic_list[0])
        # print(kind_list[0])
        # print(bisection_list[0])


    # Create iterator object
    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __repr__(self):
        return f'An object of class Result\n' \
               f'{self.benchmark = }\n' \
               f'{self.approximate_benchmark = }\n' \
               f'{self.experiment = }\n' \
               f'{self.strategy = }\n' \
               f'{self.metric = }\n'

    def __len__(self):
        return len(self.__dict__)
