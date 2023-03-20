import re
import subprocess
import os
import shutil
from .config.path import *
from .config.config import *


def get_pure_name(file_name: str) -> str:
    if file_name is None:
        return file_name
    name = file_name
    if re.search('/', file_name):
        name = file_name.split('/')[-1]
    if re.search('\.', name):
        name = name.split('.')[0]
    return name


def fix_direction(file_name: str) -> str:
    file_name = get_pure_name(file_name)
    folder, extension = OUTPUT_PATH['dot']
    dot_in_path = f'{folder}/{file_name}.{extension}'
    folder, extension = OUTPUT_PATH['gv']
    gv_out_path = f'{folder}/{file_name}.{extension}'

    dot_command = f'{DOT} {dot_in_path} -Grankdir=TB -o {gv_out_path}'
    subprocess.call([dot_command], shell=True)


def convert_verilog_to_gv(file_name: str) -> None:
    file_name = get_pure_name(file_name)

    folder, extension = OUTPUT_PATH['ver']
    verilog_in_path = f'{folder}/{file_name}.{extension}'

    folder, extension = OUTPUT_PATH['gv']
    gv_out_path = f'{folder}/{file_name}.{extension}'

    yosys_command = f"""
        read_verilog {verilog_in_path}
        show -prefix {gv_out_path[:-3]} -format gv
        """
    folder, extension = LOG_PATH['yosys']
    log_out_path = f'{folder}/{file_name}_{YOSYS}_v2gv.{extension}'
    with open(f'{log_out_path}', 'w') as y:
        subprocess.call([YOSYS, '-p', yosys_command], stdout=y)
    fix_direction(file_name)


def setup_folder_structure():
    # Setting up the folder structure
    directories = [OUTPUT_PATH['ver'][0], OUTPUT_PATH['aig'][0], OUTPUT_PATH['gv'][0], OUTPUT_PATH['z3'][0],
                   OUTPUT_PATH['report'][0], OUTPUT_PATH['figure'][0], TEST_PATH['tb'][0], TEST_PATH['z3'][0],
                   INPUT_PATH['ver'][0], INPUT_PATH['gv'][0],
                   LOG_PATH['yosys'][0]]

    for directory in directories:
        if ~os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

def clean_all():
    """
    Cleans the contents of output and test (alongside their subfolders)
    :return: Nothing
    """
    shutil.rmtree('output/')
    shutil.rmtree('test/')

