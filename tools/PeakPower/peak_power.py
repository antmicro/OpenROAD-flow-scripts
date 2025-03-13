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

    def __init__(self):
        self.header_contents = ''
        self.current_clock_time = 0
        self.clock_emits_list = []
        self.current_clock_emits = None

    def parse(self, vcd_file_path):
        with open(vcd_file_path, 'r', encoding="utf8") as vcd_file:
            reading_header = True
            
            for line in vcd_file:
                match = re.match(r'#(\d+)', line)
                if match:
                    reading_header = False
                    if self.current_clock_emits is not None:
                        self.clock_emits_list.append(self.current_clock_emits)
                    
                    self.current_clock_emits = ClockEmits()
                    self.current_clock_emits.clock_time = match.group(1)

                if reading_header:
                    self.header_contents += line
                    continue

                # clock emits
                self.current_clock_emits.clock_emits += line


parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""""")
parser.add_argument('--vcd', action='store', help='VCD file')
parser.set_defaults(stop=True)
args = parser.parse_args()

vcd = VCDFile()
vcd.parse(args.vcd)

print(vcd.header_contents)

for clock_emits in vcd.clock_emits_list:
    print(f'Clock emits from clock {clock_emits.clock_time}')
    print(clock_emits.clock_emits)