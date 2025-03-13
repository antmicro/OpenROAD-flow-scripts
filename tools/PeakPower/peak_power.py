#!/usr/bin/env python3

# Copyright 2025 Antmicro
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
import subprocess

class ClockEmits:
    clock_time: int
    clock_emits: str

    def __init__(self):
        self.clock_time = 0
        self.clock_emits = ''

class VCDFile:
    header_contents: str
    current_clock_time: int
    clock_emits_list: list[ClockEmits]
    initial_clock_emits: ClockEmits
    current_clock_emits = ClockEmits

    def __init__(self):
        self.header_contents = ''
        self.current_clock_time = 0
        self.clock_emits_list = []
        self.initial_clock_emits = None
        self.current_clock_emits = None

    def parse(self, vcd_file_path: str):
        with open(vcd_file_path, 'r', encoding="utf8") as vcd_file:
            reading_header = True
            
            for line in vcd_file:
                match = re.match(r'#(\d+)', line)
                if match:
                    reading_header = False

                    if self.current_clock_emits is not None and self.current_clock_emits != self.initial_clock_emits:
                        self.clock_emits_list.append(self.current_clock_emits)

                    self.current_clock_emits = ClockEmits()
                    self.current_clock_emits.clock_time = match.group(1)
                    
                    if self.initial_clock_emits is None:
                        self.initial_clock_emits = self.current_clock_emits

                if reading_header:
                    self.header_contents += line
                    continue

                # clock emits
                self.current_clock_emits.clock_emits += line


class VCDIterator:
    current_clock_emits_index: int
    source_vcd_file: VCDFile

    def __init__(self, vcd_file: VCDFile):
        self.current_clock_emits_index = 0
        self.source_vcd_file = vcd_file

    def next_vcd_clock_chunk_available(self):
        return self.current_clock_emits_index < len(self.source_vcd_file.clock_emits_list)

    def create_next_vcd_clock_chunk(self):
        vcd_contents = self.source_vcd_file.header_contents
        vcd_contents += self.source_vcd_file.initial_clock_emits.clock_emits
        vcd_contents += self.source_vcd_file.clock_emits_list[self.current_clock_emits_index].clock_emits

        self.current_clock_emits_index += 1

        return vcd_contents
    

def save_to_file(file_contents: str, file_path: str):
    with open(file_path, 'w') as file:
        file.write(file_contents)


def read_from_file(file_name: str):
    with open(file_name, 'r', encoding="utf8") as file:
        return file.readlines()


def search_for_total_power(report_power: list[str]):
    for line in report_power:
        match = re.match(r'Total\s+(([\w\.\-]+\s+)+)', line)
        if match:
            total_power = match.group(2).replace(' ', '')
            print(f'Total power: {total_power} Watts')
            return float(total_power)
        
    return 0


parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""""")
parser.add_argument('--vcd', action='store', help='VCD file')
parser.set_defaults(stop=True)
args = parser.parse_args()

scratch_file_name = 'temp.vcd'
open_sta_command = 'sta'
open_sta_script = 'power_vcd.tcl'
power_report_file = 'output_power'

vcd = VCDFile()
vcd.parse(args.vcd)

vcd_iterator = VCDIterator(vcd)

peak_power = 0

while vcd_iterator.next_vcd_clock_chunk_available():
    vcd_contents = vcd_iterator.create_next_vcd_clock_chunk()
    save_to_file(vcd_contents, scratch_file_name)
    subprocess.run([open_sta_command, open_sta_script], capture_output=True, text=True)
    report_contents = read_from_file(power_report_file)
    total_power = search_for_total_power(report_contents)
    peak_power = max(peak_power, total_power)
    print(f"Peak power: {peak_power} Watts")