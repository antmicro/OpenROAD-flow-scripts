#!/usr/bin/env python

import unittest
import subprocess
import pathlib
import os
import multiprocessing

PLATFORM = "asap7"

VERILATOR_ROOT = os.environ.get('VERILATOR_ROOT')
QUESTA_BIN = os.environ.get('QUESTA_BIN')
LM_LICENSE_FILE = os.environ.get("LM_LICENSE_FILE")

ORFS_ROOT = pathlib.Path(__file__).parent.parent.resolve()
CPUS = multiprocessing.cpu_count()

BASE_DIR = f"{ORFS_ROOT}/flow"
STDCELL_DIR = f"{BASE_DIR}/platforms/{PLATFORM}/verilog/stdcell"
YOSYS_WORKAROUND_DIR = f"{BASE_DIR}/platforms/{PLATFORM}/work_around_yosys"
YOSYS_WORKAROUND_SRCS = f"{YOSYS_WORKAROUND_DIR}/asap7sc7p5t_OA_RVT_TT_201020.v"
STDCELL_SRC = f"{STDCELL_DIR}/asap7sc7p5t_AO_RVT_TT_201020.v {STDCELL_DIR}/asap7sc7p5t_INVBUF_RVT_TT_201020.v {STDCELL_DIR}/asap7sc7p5t_SEQ_RVT_TT_220101.v {STDCELL_DIR}/asap7sc7p5t_SIMPLE_RVT_TT_201020.v"
SRCS = f"{STDCELL_SRC} {YOSYS_WORKAROUND_SRCS}"


class TestAes(unittest.TestCase):
    DESIGN_NAME = "aes"
    DESIGN_CONFIG = f"designs/{PLATFORM}/{DESIGN_NAME}/config.mk"
    RESULTS_DIR = f"{BASE_DIR}/results/{PLATFORM}/{DESIGN_NAME}"
    MAIN_SRC = f"{RESULTS_DIR}/base/1_synth.v"
    TEST_BENCH = f"{ORFS_ROOT}/bench/{DESIGN_NAME}/test_bench_top.v"
    AES_SRC = f"{BASE_DIR}/designs/src/{DESIGN_NAME}"
    FILES = f"{SRCS} {MAIN_SRC} {TEST_BENCH} {AES_SRC}/aes_inv_cipher_top.v {AES_SRC}/aes_inv_sbox.v {AES_SRC}/aes_key_expand_128.v {AES_SRC}/aes_rcon.v {AES_SRC}/aes_sbox.v"

    @classmethod
    def setUpClass(self):
        subprocess.call(
            f"make -C {ORFS_ROOT}/flow DESIGN_CONFIG={self.DESIGN_CONFIG} synth",
            shell=True)

    @unittest.skipIf(VERILATOR_ROOT is None, "Requires VERILATOR_ROOT defined")
    def test_verilator(self):
        self.assertFalse(
            subprocess.call(
                f"{VERILATOR_ROOT}/bin/verilator -Mdir {self.RESULTS_DIR}/verilator --binary --error-limit 9999999 -Wno-MULTIDRIVEN -Wno-NOLATCH {self.FILES} +incdir+{self.AES_SRC} -j {CPUS} --build-jobs {CPUS}",
                shell=True))
        self.assertFalse(
            subprocess.call(
                f"{self.RESULTS_DIR}/verilator/Vasap7sc7p5t_AO_RVT_TT_201020",
                shell=True))

    @unittest.skipIf(QUESTA_BIN is None or LM_LICENSE_FILE is None,
                     "Requires QUESTA_BIN and LM_LICENSE_FILE defined")
    def test_questa(self):
        subprocess.call(
            f"{QUESTA_BIN} {self.FILES} +incdir+{self.AES_SRC} -outdir {self.RESULTS_DIR}/questa",
            shell=True,
            env=dict(os.environ, LM_LICENSE_FILE=LM_LICENSE_FILE))

    @classmethod
    def tearDownClass(self):
        subprocess.call(
            f"make -C {ORFS_ROOT}/flow DESIGN_CONFIG={self.DESIGN_CONFIG} clean_synth",
            shell=True)


def run_cmd(cmd):
    subprocess.run([cmd])


if __name__ == "__main__":
    unittest.main()
