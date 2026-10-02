from setuptools import setup
import os

lib_dir = "../build/lib"
pycupdlp_lib = [
    lib_dir + "/" + i for i in os.listdir(lib_dir) if i.startswith("pycupdlp")
][0]

setup(
    name="pycupdlp",
    version="1.0",
    author=(
        "Haihao Lu, Jinwen Yang, Haodong Hu, Qi Huangfu, Jinsong Liu, "
        "Tianhao Liu, Yinyu Ye, Chuwen Zhang, Dongdong Ge"
    ),
    data_files=[pycupdlp_lib],
)
