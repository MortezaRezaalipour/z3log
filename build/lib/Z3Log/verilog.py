import os
import regex as re
import random
import numpy as np
import subprocess
import sys
from .config.path import *
from .config.config import *
from .argument import Arguments
from .utils import *
from typing import Dict, List
from colorama import Fore, Style
from collections import OrderedDict

class Verilog:
    def __init__(self, benchmark_name: str, samples: list = []):
        """
        takes in a circuit and creates a verilog object out of it
        :param benchmark_name: the input circuit in verilog format
        :param samples: number of samples for the mc simulation; it is an empty list by default
        """
        self.__circuit_name = get_pure_name(benchmark_name)

        folder, extension = INPUT_PATH['ver']
        self.__circuit_in_path = f'{folder}/{benchmark_name}.{extension}'

        folder, extension = OUTPUT_PATH['ver']
        self.__circuit_out_path = f'{folder}/{benchmark_name}.{extension}'

        folder, extension = OUTPUT_PATH['aig']
        self.__aig_out_path = f'{folder}/{benchmark_name}.{extension}'

        folder, extension = TEST_PATH['tb']
        self.__testbench_out_path = f'{folder}/{benchmark_name}_tb.{extension}'

        folder, extension = TEST_PATH['tb']
        self.__tmp_verilog = f'{folder}/{TMP}.{extension}'

        folder, extension = TEST_PATH['iver']
        self.__iverilog_out_path = f'{folder}/{benchmark_name}.{extension}'

        folder, extension = LOG_PATH['iver']
        self.__iverilog_log_path = f'{folder}/{benchmark_name}_{IVERILOG}_{LOG}.{extension}'

        folder, extension = TEST_PATH['vvp']
        self.__vvp_out_path = f'{folder}/{benchmark_name}.{extension}'

        self.__num_inputs, self.__num_outputs = self.extract_module_io()


        self.__sample_results = None
        self.__samples = samples

        self.synthesize_to_gate_level(self.in_path, self.out_path)
        # self.unwrap_variables()


    @property
    def name(self):
        return self.__circuit_name

    @name.setter
    def name(self, newname):
        self.__circuit_name == newname
    @property
    def num_inputs(self):
        return self.__num_inputs

    @property
    def num_outputs(self):
        return self.__num_outputs

    @property
    def in_path(self):
        return self.__circuit_in_path

    @property
    def out_path(self):
        return self.__circuit_out_path

    @property
    def aig_out_path(self):
        return self.__aig_out_path

    @property
    def cleaned_verilog(self):
        return self.__circuit_out_path

    @property
    def tmp(self):
        return self.__tmp_verilog

    @property
    def testbench(self):
        return self.__testbench_out_path

    @property
    def samples(self):
        return self.__samples

    @property
    def iverilog_out_path(self):
        return self.__iverilog_out_path

    @property
    def iverilog_log_path(self):
        return self.__iverilog_log_path

    @property
    def vvp_in_path(self):
        return self.__iverilog_out_path

    @property
    def vvp_out_path(self):
        return self.__vvp_out_path

    def set_samples(self, samples: np.array or list):
        self.__samples = samples

    @property
    def sample_results(self):
        return self.__sample_results

    def set_sample_results(self, results):
        self.__sample_results = results

    # methods
    def import_results(self):
        arr = []
        with open(self.vvp_out_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                arr.append(int(line, 2))
        results = np.array(arr)

        return results

    def import_circuit(self):
        with open(self.in_path, 'r') as f:
            lines = f.readlines()
        return lines

    def check_multi_vector_declaration(self, cur_list: List[str]) -> List[str]:
        # for lines such as:
        # input [1:0]a, b;
        # cur_list will be = ['[1:0]a', 'b'] which should be ['[1:0]a', '[1:0]b']
        if self.is_vector(cur_list[0]):
            # find the range
            vector_range = re.search('\[\d+:\d+\]', cur_list[0]).group()
            # propagate the range for the rest of the elements of cur_list
            for i in range(1, len(cur_list)):
                cur_list[i] = vector_range + cur_list[i]
        return cur_list

    def is_vector(self, var_name: str) -> bool:
        """
        checks whether var_name variable is a vector
        :param var_name: the variable name
        :return: True if the variable is a vector, otherwise returns False
        """
        if re.search('(\[\d+:\d+\])', var_name):
            return True
        else:
            return False

    def get_name(self, var_name: str) -> str:
        """
        if var_name is an array, e.g., [1:0]a, it will return a. Otherwise, it will return var_name.
        :param var_name: the name of the variable
        :return: a string representing variable name
        """
        if self.is_vector(var_name):
            # remove [n:m] part
            match_obj = re.search('(\[\d+:\d+\])(.*)', var_name)
            return match_obj.group(2)
        else:
            return var_name

    def get_width(self, var_name: str) -> int:
        """
        computes the bit-width of a given variable
        :param var_name: the name of the variable
        :return: an integer representing the bit-width of the given variable
        """
        if self.is_vector(var_name):
            # compute the length
            match = re.search('\[(\d+):(\d+)\]', var_name)  # [1:0]a
            l_bound = int(match.group(1))  # 1
            r_bound = int(match.group(2))  # 0
            return abs((l_bound - r_bound) + 1)
        else:
            return 1

    def extract_inputs_outputs(self, verilog_str: List[str], port_list: List[str]):
        input_dict: Dict[int:(str, int)] = {}
        output_dict: Dict[int:(str, int)] = {}
        # example:
        # for module circuit(a, b, c, d)
        # input [1:0] a;
        # input [2:0]b;
        # output d;
        # output [3:0]c;
        # example input_dict = {0:(a, 2), 1:(b, 3)}
        # example input_dict = {3:(d, 1), 2:(c, 3)}
        for line in verilog_str:
            line = line.strip()  # remove all whitespaces at the beginning or end of the line
            # extract inputs

            if line.startswith('input'):
                match_obj = re.search('input (.+)', line)  # input a, b, c; or input [1:0] a;
                cur_input_str = match_obj.group(1)  # a, b, c or [1:0] a
                cur_input_list = cur_input_str.strip().replace(" ", "").split(',')  # ['a', 'b', 'c'] or ['[1:0]a']
                cur_input_list = self.check_multi_vector_declaration(cur_input_list)
                for inp in cur_input_list:
                    if self.get_name(inp) in port_list:
                        position_in_module_signature = port_list.index(self.get_name(inp))
                        input_dict[position_in_module_signature] = (self.get_name(inp), self.get_width(inp))
                    else:
                        raise Exception(f"input name {self.get_name(inp)} is not in the port_list")
            # extract outputs
            if line.startswith('output'):
                match_obj = re.search('output (.+)', line)  # input a, b, c; or input [1:0] a;
                cur_output_str = match_obj.group(1)  # a, b, c or [1:0] a
                cur_output_list = cur_output_str.strip().replace(" ", "").split(',')  # ['a', 'b', 'c'] or ['[1:0]a']
                cur_output_list = self.check_multi_vector_declaration(cur_output_list)
                for out in cur_output_list:
                    if self.get_name(out) in port_list:
                        position_in_module_signature = port_list.index(self.get_name(out))
                        output_dict[position_in_module_signature] = (self.get_name(out), self.get_width(out))
                    else:
                        raise Exception(f"output name {self.get_name(out)} is not in the port_list")

            sorted_input_dict = OrderedDict(sorted(input_dict.items()))
            sorted_output_dict = OrderedDict(sorted(output_dict.items()))
        return sorted_input_dict, sorted_output_dict

    def extract_module_signature(self, verilog_str: List[str]):
        """
        reads the first line of a Verilog netlist and extracts module and port names
        :param verilog_str1: the Verilog description as a string
        :return: modulename as a string variable and port names as a list
        """
        module_name = None
        port_list = None


        for line in verilog_str:
            line = line.strip()  # remove whitespaces at the beginning or end of the line

            if re.search('module', line) and not re.search('endmodule', line):
                # extract module

                match_object = re.search('module (\w+)', line)  # module adder(a, b, c)
                module_name = match_object.group(1)  # adder

                # extract port list

                line = line.split(module_name)[1].replace("\n", "").strip()

                match_object = re.search('\((.+)\)', line)  # module adder(a, b, c)

                ports_str = match_object.group(1)  # a, b, c

                port_list = ports_str.strip().replace(" ", "").split(',')
        assert module_name and port_list, f'Either module_name or port_list is None'

        return module_name, port_list


    def synthesize_to_gate_level(self, input_path: str, output_path: str):
        yosys_command = f"""
        read_verilog {input_path};
        synth -flatten;
        opt;
        opt_clean -purge;
        abc -g NAND;
        opt;
        opt_clean -purge;
        splitnets -ports;
        opt;
        opt_clean -purge;
        write_verilog -noattr {output_path};
        """
        process = subprocess.run(['yosys', '-p', yosys_command], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if process.stderr.decode():
            print(f'Error!')
            raise Exception(f'ERROR!!! yosys cannot do its pass on file {input_path}\n{process.stderr.decode()}')

        self.rename_variables(output_path, output_path)



    def rename_variables(self, input_path: str, output_path: str):
        with open(f'{input_path}', 'r') as infile:
            verilog_str = infile.read()

        # print(f'{verilog_str = }')

        verilog_str_tmp = verilog_str.split(';')
        verilog_str = verilog_str.split('\n')

        # print(f'{verilog_str = }')

        module_name, port_list = self.extract_module_signature(verilog_str_tmp)
        # print(f'{module_name = }')
        # print(f'{port_list = }')

        input_dict, output_dict = self.extract_inputs_outputs(verilog_str_tmp, port_list)
        # print(f'{input_dict = }')
        # print(f'{output_dict = }')

        new_labels = self.create_new_labels(port_list, input_dict, output_dict)
        # print(f'{new_labels = }')





        verilog_str = self.relabel_nodes(verilog_str, new_labels)

        with open(f'{output_path}', 'w') as outfile:
            for line in verilog_str:
                outfile.write(f'{line}\n')

    def create_new_labels(self, port_list: List, input_dict: Dict, output_dict: Dict):
        new_labels: Dict = {}
        for port_idx in input_dict:
            if input_dict[port_idx][0] == port_list[port_idx]:
                new_labels[port_list[port_idx]] = f'in{port_idx}'
                # print(f'{new_labels = }')
            else:
                raise Exception(f'Error!!! {input_dict[port_idx][0]} is not equal to {port_list[port_idx]}')

        out_idx = 0
        for port_idx in output_dict:
            if output_dict[port_idx][0] == port_list[port_idx]:
                # print(Fore.RED + f'{port_idx = }' + Style.RESET_ALL)
                new_labels[port_list[port_idx]] = f'out{out_idx}'
                out_idx += 1
                # print(f'{new_labels = }')
            else:
                raise Exception(f'Error!!! {output_dict[port_idx][0]} is not equal to {port_list[port_idx]}')
        return new_labels

    def relabel_nodes(self, verilog_str: List[str], new_labels: Dict):
        verilog_str_tmp = verilog_str

        for line_idx, line in enumerate(verilog_str):
            for key, value in new_labels.items():

                escaped_key = re.escape(key)
                if re.search(f'{escaped_key}[,;)\s\n\r]|({escaped_key})$', line):
                    found = re.search(f'({escaped_key})[,;)\s\r\n]|({escaped_key})$', line)
                    middle = found.group(1)
                    end = found.group(2)

                    s = found.span()[0]
                    if found.group(1):
                        e = s + len(found.group(1))
                        line = f"{line[:s]}{value}{line[e:]}"
                    elif found.group(2):
                        line = f"{line[:s]}{value}"
                    else:
                        print(Fore.RED + f'ERROR! in (__name__): variable{key} does not belong in either of the two groups!'+ Style.RESET_ALL)
                    # print(f'{line  =}')
                    verilog_str_tmp[line_idx] = line
                # if key in line:
                #     print(f'{key = }')
                #     print(f'{line = }')
                #     exit()
        return verilog_str_tmp

    # Deprecated!
    # def synthesize_to_gate_level(self):
    #     yosys_command = f"""
    #     read_verilog {self.in_path}
    #     synth -flatten
    #     opt
    #     opt_clean -purge
    #     abc -g AND,OR
    #     write_verilog -noattr {self.out_path}
    #     abc -g NAND;
    #     aigmap;
    #     opt;
    #     opt_clean -purge;
    #     write_aiger {self.aig_out_path};
    #     """
    #     with open('log.txt', 'w') as y:
    #         subprocess.call([YOSYS, '-p', yosys_command])


    def unwrap_variables(self) -> None:
        """

        :return:
        """
        lsoracle_command = f"""
        read_aig {self.aig_out_path};
        write_verilog {self.out_path}
        """
        with open('lsoracle_log.txt', 'w') as f:
            subprocess.call([LSORACLE, '-c', lsoracle_command])
        self.fix_module_name()

    def fix_module_name(self):
        with open(self.out_path, 'r') as f:
            lines = f.readlines()
        with open(self.out_path, 'w') as g:
            for idx, line in enumerate(lines):
                if re.search('top', line):
                    line = line.replace('top', self.name)
                    lines[idx] = line
                    break
            for line in lines:
                g.write(line)

    def create_test_bench(self, samples: 'list[int]' = []):

        # read a clean verilog
        # create a test bench for it
        num = len(self.samples)
        with open(self.testbench, 'w') as f:
            modulename, port_list, inp, n_inputs, out, n_outputs = self.extract_module_info()

            f.write("module " + modulename + "_tb;\n")
            f.write('reg [' + str(n_inputs - 1) + ':0] pi;\n')
            f.write('wire [' + str(n_outputs - 1) + ':0] po;\n')
            f.write(modulename + ' dut(')

            first = True
            inp_count = 0
            out_count = 0

            for i in port_list:
                if not first:
                    f.write(',')
                first = False

                if i in inp:
                    if inp[i] > 1:
                        f.write(' pi[{}:{}] '.format(inp_count + inp[i] - 1, inp_count))
                    else:
                        f.write(' pi[{}] '.format(inp_count))
                    inp_count += inp[i]

                elif i in out:
                    if out[i] > 1:
                        f.write(' po[{}:{}] '.format(out_count + out[i] - 1, out_count))
                    else:
                        f.write(' po[{}] '.format(out_count))
                    out_count += out[i]

                else:
                    print('[Error] Port {} is not defined as input or output.'.format(i))
                    sys.exit(0)

            f.write(');\n')

            f.write("initial\n")
            f.write("begin\n")

            for idx, sample in enumerate(self.samples):

                f.write('# 1  pi=' + str(n_inputs) + '\'b')
                f.write('{0:0>{1}}'.format(str(bin(sample))[2:], n_inputs))

                f.write(';\n')
                f.write("#1 $display(\"%b\", po);\n")

            f.write("end\n")
            f.write("endmodule\n")

    def extract_module_io(self) -> (int, int):
        clean_verilog = self.in_path
        yosys_command = 'read_verilog ' + clean_verilog + '; synth -flatten; opt; opt_clean; techmap; write_verilog ' + self.tmp + ';\n'
        process = subprocess.run([YOSYS, '-p', yosys_command], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if process.stderr.decode():
            print(f'Error!')
            raise Exception(f'ERROR!!! yosys cannot do its pass on file {clean_verilog}\n{process.stderr.decode()}')
        with open(self.tmp) as tmp_file:
            inp = {}
            inp_count = 0
            out = {}
            out_count = 0
            modulename = None
            line = tmp_file.readline()
            while line:
                tokens = re.split('[ ()]', line.strip().strip(';').strip())

                if len(tokens) > 0 and tokens[0] == 'module' and modulename is None:
                    modulename = tokens[1]
                    port_list = re.split('[,()]', line.strip().strip(';').strip())[1:]
                    port_list = [s.strip() for s in port_list if s.strip() != '']

                if len(tokens) == 2 and (tokens[0] == 'input' or tokens[0] == 'output'):
                    if tokens[0] == 'input':
                        inp[tokens[1]] = 1
                        inp_count += 1
                    if tokens[0] == 'output':
                        out[tokens[1]] = 1
                        out_count += 1

                if len(tokens) == 3 and (tokens[0] == 'input' or tokens[0] == 'output'):
                    range_str = tokens[1][1:-1].split(':')
                    range_int = list(map(int, range_str))
                    length = max(range_int) - min(range_int) + 1
                    if tokens[0] == 'input':
                        inp[tokens[2]] = length
                        inp_count += length
                    if tokens[0] == 'output':
                        out[tokens[2]] = length
                        out_count += length

                line = tmp_file.readline()

        os.remove(self.tmp)

        return inp_count, out_count

    def extract_module_info(self) -> 'tuple(str, list(str), dict, int, dict, int)':
        """
        reads a verilog file and extracts the signature
        :return: a tuple containing the modulename, the list of input and output names, two dictionary that hold the bitwidth
        of all inputs and outputs, and the number of inputs and outputs
        """
        clean_verilog = self.out_path
        yosys_command = 'read_verilog ' + clean_verilog + '; synth -flatten; opt; opt_clean; techmap; write_verilog ' + self.tmp + ';\n'
        subprocess.call([YOSYS, '-p', yosys_command], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        with open(self.tmp) as tmp_file:
            inp = {}
            inp_count = 0
            out = {}
            out_count = 0
            modulename = None
            line = tmp_file.readline()
            while line:
                tokens = re.split('[ ()]', line.strip().strip(';').strip())

                if len(tokens) > 0 and tokens[0] == 'module' and modulename is None:
                    modulename = tokens[1]
                    port_list = re.split('[,()]', line.strip().strip(';').strip())[1:]
                    port_list = [s.strip() for s in port_list if s.strip() != '']

                if len(tokens) == 2 and (tokens[0] == 'input' or tokens[0] == 'output'):
                    if tokens[0] == 'input':
                        inp[tokens[1]] = 1
                        inp_count += 1
                    if tokens[0] == 'output':
                        out[tokens[1]] = 1
                        out_count += 1

                if len(tokens) == 3 and (tokens[0] == 'input' or tokens[0] == 'output'):
                    range_str = tokens[1][1:-1].split(':')
                    range_int = list(map(int, range_str))
                    length = max(range_int) - min(range_int) + 1
                    if tokens[0] == 'input':
                        inp[tokens[2]] = length
                        inp_count += length
                    if tokens[0] == 'output':
                        out[tokens[2]] = length
                        out_count += length

                line = tmp_file.readline()

        os.remove(self.tmp)

        return modulename, port_list, inp, inp_count, out, out_count

    def run_test_bench(self):
        iverilog_command = f'{IVERILOG} -o {self.iverilog_out_path} ' \
                           f'{self.cleaned_verilog} ' \
                           f'{self.testbench} '

        vvp_command = f'{self.iverilog_out_path}'

        with open(f'{self.iverilog_log_path}', 'w') as f:
            subprocess.call(iverilog_command, shell=True, stdout=f)

        with open(f'{self.vvp_out_path}', 'w') as f:
            subprocess.call([VVP, vvp_command], stdout=f)

        os.remove(f'{self.iverilog_out_path}')

        self.set_sample_results(self.import_results())

    def export_circuit(self):
        self.synthesize_to_gate_level(self.in_path, self.out_path)
        # self.unwrap_variables()

    def __repr__(self):
        return f'An object of class Verilog\n' \
               f'{self.name = }\n' \
               f'{self.in_path = }\n' \
               f'{self.out_path = }\n'
