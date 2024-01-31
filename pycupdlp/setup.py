from setuptools import setup
import os

lib_dir = "../build/lib"
pycupdlp_lib = [
    lib_dir + "/" + i for i in os.listdir(lib_dir) if i.startswith("pycupdlp")
][0]

setup(
    name="pycupdlp",
    version="1.0",
    author="Jinsong Liu, Tianhao Liu, Chuwen Zhang",
    data_files=[pycupdlp_lib],
)
