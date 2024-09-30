import argparse
import glob
import json
import os
import re
import sys
from subprocess import run

import numpy as np

SDC_TEMPLATE = """
set clk_name  core_clock
set clk_port_name clk
set clk_period 2000
set clk_io_pct 0.2

set clk_port [get_ports $clk_port_name]

create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]

set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]
"""
STAGE_TO_METRICS = {
    "route": "detailedroute",
    "place": "detailedplace",
    "final": "finish",
}

def write_sdc(variables, path, sdc_original, constraints_sdc):
    """
    Create a SDC file with parameters for current tuning iteration.
    """
    # TODO: handle case where the reference file does not exist
    new_file = sdc_original
    for key, value in variables.items():
        if key == "CLK_PERIOD":
            if new_file.find("set clk_period") != -1:
                new_file = re.sub(
                    r"set clk_period .*\n(.*)", f"set clk_period {value}\n\\1", new_file
                )
            else:
                new_file = re.sub(
                    r"-period [0-9\.]+ (.*)", f"-period {value} \\1", new_file
                )
                new_file = re.sub(r"-waveform [{}\s0-9\.]+[\s|\n]", "", new_file)
        elif key == "UNCERTAINTY":
            if new_file.find("set uncertainty") != -1:
                new_file = re.sub(
                    r"set uncertainty .*\n(.*)",
                    f"set uncertainty {value}\n\\1",
                    new_file,
                )
            else:
                new_file += f"\nset uncertainty {value}\n"
        elif key == "IO_DELAY":
            if new_file.find("set io_delay") != -1:
                new_file = re.sub(
                    r"set io_delay .*\n(.*)", f"set io_delay {value}\n\\1", new_file
                )
            else:
                new_file += f"\nset io_delay {value}\n"
    file_name = path + f"/{constraints_sdc}"
    with open(file_name, "w") as file:
        file.write(new_file)
    return file_name


def write_fast_route(variables, path, fr_original, fastroute_tcl):
    """
    Create a FastRoute Tcl file with parameters for current tuning iteration.
    """
    # TODO: handle case where the reference file does not exist
    layer_cmd = "set_global_routing_layer_adjustment"
    new_file = fr_original
    for key, value in variables.items():
        if key.startswith("LAYER_ADJUST"):
            layer = key.lstrip("LAYER_ADJUST")
            # If there is no suffix (i.e., layer name) apply adjust to all
            # layers.
            if layer == "":
                new_file += "\nset_global_routing_layer_adjustment"
                new_file += " $::env(MIN_ROUTING_LAYER)"
                new_file += "-$::env(MAX_ROUTING_LAYER)"
                new_file += f" {value}"
            elif re.search(f"{layer_cmd}.*{layer}", new_file):
                new_file = re.sub(
                    f"({layer_cmd}.*{layer}).*\n(.*)", f"\\1 {value}\n\\2", new_file
                )
            else:
                new_file += f"\n{layer_cmd} {layer} {value}\n"
        elif key == "GR_SEED":
            new_file += f"\nset_global_routing_random -seed {value}\n"
    file_name = path + f"/{fastroute_tcl}"
    with open(file_name, "w") as file:
        file.write(new_file)
    return file_name


def parse_flow_variables(platform):
    """
    Parse the flow variables from source
    - Code: Makefile `vars` target output

    TODO: Tests.

    Output:
    - flow_variables: set of flow variables
    """
    cur_path = os.path.dirname(os.path.realpath(__file__))

    # first, generate vars.tcl
    makefile_path = os.path.join(cur_path, "../../../../flow/")
    initial_path = os.path.abspath(os.getcwd())
    os.chdir(makefile_path)
    result = run(["make", "vars", f"PLATFORM={platform}"])
    if result.returncode != 0:
        print(f"[ERROR TUN-0018] Makefile failed with error code {result.returncode}.")
        sys.exit(1)
    if not os.path.exists("vars.tcl"):
        print("[ERROR TUN-0019] Makefile did not generate vars.tcl.")
        sys.exit(1)
    os.chdir(initial_path)

    # for code parsing, you need to parse from both scripts and vars.tcl file.
    pattern = r"(?:::)?env\((.*?)\)"
    files = glob.glob(os.path.join(cur_path, "../../../../flow/scripts/*.tcl"))
    files.append(os.path.join(cur_path, "../../../../flow/vars.tcl"))
    variables = set()
    for file in files:
        with open(file) as fp:
            matches = re.findall(pattern, fp.read())
        for match in matches:
            for variable in match.split("\n"):
                variables.add(variable.strip().upper())
    return variables


def parse_config(
    config,
    platform,
    sdc_original,
    constraints_sdc,
    fr_original,
    fastroute_tcl,
    path=os.getcwd(),
):
    """
    Parse configuration received from tune into make variables.
    """
    options = ""
    sdc = {}
    fast_route = {}
    flow_variables = parse_flow_variables(platform)
    for key, value in config.items():
        # Keys that begin with underscore need special handling.
        if key.startswith("_"):
            # Variables to be injected into fastroute.tcl
            if key.startswith("_FR_"):
                fast_route[key.replace("_FR_", "", 1)] = value
            # Variables to be injected into constraints.sdc
            elif key.startswith("_SDC_"):
                sdc[key.replace("_SDC_", "", 1)] = value
            # Special substitution cases
            elif key == "_PINS_DISTANCE":
                options += f' PLACE_PINS_ARGS="-min_distance {value}"'
            elif key == "_SYNTH_FLATTEN":
                print(
                    "[WARNING TUN-0013] Non-flatten the designs are not "
                    "fully supported, ignoring _SYNTH_FLATTEN parameter."
                )
        # Default case is VAR=VALUE
        else:
            # Sanity check: ignore all flow variables that are not tunable
            if key not in flow_variables:
                print(f"[ERROR TUN-0017] Variable {key} is not tunable.")
                sys.exit(1)
            options += f" {key}={value}"
    if bool(sdc):
        write_sdc(sdc, path, sdc_original, constraints_sdc)
        options += f" SDC_FILE={path}/{constraints_sdc}"
    if bool(fast_route):
        write_fast_route(fast_route, path, fr_original, fastroute_tcl)
        options += f" FASTROUTE_TCL={path}/{fastroute_tcl}"
    return options


def run_command(
    args, cmd, timeout=None, stderr_file=None, stdout_file=None, fail_fast=False
):
    """
    Wrapper for subprocess.run
    Allows to run shell command, control print and exceptions.
    """
    process = run(
        cmd, timeout=timeout, capture_output=True, text=True, check=False, shell=True
    )
    if stderr_file is not None and process.stderr != "":
        with open(stderr_file, "a") as file:
            file.write(f"\n\n{cmd}\n{process.stderr}")
    if stdout_file is not None and process.stdout != "":
        with open(stdout_file, "a") as file:
            file.write(f"\n\n{cmd}\n{process.stdout}")
    if args.verbose >= 1:
        print(process.stderr)
    if args.verbose >= 2:
        print(process.stdout)

    if fail_fast and process.returncode != 0:
        raise RuntimeError


def openroad(
    args,
    base_dir,
    parameters,
    flow_variant,
    path="",
    install_path=os.path.abspath("../tools/install"),
    stage="",
):
    """
    Run OpenROAD-flow-scripts with a given set of parameters.
    """
    # Make sure path ends in a slash, i.e., is a folder
    flow_variant = f"{args.experiment}/{flow_variant}"
    if path != "":
        log_path = f"{path}/{flow_variant}/"
        report_path = log_path.replace("logs", "reports")
        run_command(args, f"mkdir -p {log_path}")
        run_command(args, f"mkdir -p {report_path}")
    else:
        log_path = report_path = os.getcwd() + "/"

    export_command = f"export PATH={install_path}/OpenROAD/bin"
    export_command += f":{install_path}/yosys/bin:$PATH"
    export_command += " && "

    make_command = export_command
    make_command += f"make -C {base_dir}/flow DESIGN_CONFIG=designs/"
    make_command += f"{args.platform}/{args.design}/config.mk {stage}"
    make_command += f" PLATFORM={args.platform}"
    make_command += f" FLOW_VARIANT={flow_variant} {parameters}"
    make_command += " EQUIVALENCE_CHECK=0"
    make_command += f" NPROC={args.openroad_threads} SHELL=bash"
    run_command(
        args,
        make_command,
        timeout=args.timeout,
        stderr_file=f"{log_path}error-make-finish.log",
        stdout_file=f"{log_path}make-finish-stdout.log",
    )

    metrics_file = os.path.join(report_path, "metrics.json")
    metrics_command = export_command
    metrics_command += f"{base_dir}/flow/util/genMetrics.py -x"
    metrics_command += f" -v {flow_variant}"
    metrics_command += f" -d {args.design}"
    metrics_command += f" -p {args.platform}"
    metrics_command += f" -o {metrics_file}"
    run_command(
        args,
        metrics_command,
        stderr_file=f"{log_path}error-metrics.log",
        stdout_file=f"{log_path}metrics-stdout.log",
    )

    return metrics_file


STAGES = list(
    enumerate(
        [
            "synth",
            "floorplan",
            "floorplan_io",
            "floorplan_tdms",
            "floorplan_macro",
            "floorplan_tap",
            "floorplan_pdn",
            "globalplace",
            "detailedplace",
            "cts",
            "globalroute",
            "detailedroute",
        ]
    )
)


def read_metrics(file_name, stage=""):
    """
    Collects metrics to evaluate the user-defined objective function.
    """
    metric_name = STAGE_TO_METRICS.get(stage if stage else "final", stage)
    with open(file_name) as file:
        data = json.load(file)
    clk_period = 9999999
    worst_slack = "ERR"
    wirelength = "ERR"
    num_drc = "ERR"
    total_power = "ERR"
    core_util = "ERR"
    final_util = "ERR"
    design_area = "ERR"
    die_area = "ERR"
    core_area = "ERR"
    last_stage = -1
    for stage_name, value in data.items():
        if stage_name == "constraints" and len(value["clocks__details"]) > 0:
            clk_period = float(value["clocks__details"][0].split()[1])
        if stage_name == "floorplan" and "design__instance__utilization" in value:
            core_util = value["design__instance__utilization"]
        if stage_name == "detailedroute" and "route__drc_errors" in value:
            num_drc = value["route__drc_errors"]
        if stage_name == "detailedroute" and "route__wirelength" in value:
            wirelength = value["route__wirelength"]
        if stage_name == metric_name and "timing__setup__ws" in value:
            worst_slack = value["timing__setup__ws"]
        if stage_name == metric_name and "power__total" in value:
            total_power = value["power__total"]
        if stage_name == metric_name and "design__instance__utilization" in value:
            final_util = value["design__instance__utilization"]
        if stage_name == metric_name and "design__instance__area" in value:
            design_area = value["design__instance__area"]
        if stage_name == metric_name and "design__core__area" in value:
            core_area = value["design__core__area"]
        if stage_name == metric_name and "design__die__area" in value:
            die_area = value["design__die__area"]
    for i, stage_name in reversed(STAGES):
        if stage_name in data and [d for d in data[stage_name].values() if d != "ERR"]:
            last_stage = i
            break
    ret = {
        "clk_period": clk_period,
        "worst_slack": worst_slack,
        "total_power": total_power,
        "core_util": core_util,
        "final_util": final_util,
        "design_area": design_area,
        "core_area": core_area,
        "die_area": die_area,
        "last_successful_stage": last_stage,
    } | ({
        "wirelength": wirelength,
        "num_drc": num_drc,
    } if metric_name in ("detailedroute", "finish") else {})
    return ret


def read_config(file_name, mode, algorithm):
    """
    Please consider inclusive, exclusive
    Most type uses [min, max)
    But, Quantization makes the upper bound inclusive.
    e.g., qrandint and qlograndint uses [min, max]
    step value is used for quantized type (e.g., quniform). Otherwise, write 0.
    When min==max, it means the constant value
    """

    def read(path):
        with open(os.path.abspath(path), "r") as file:
            ret = file.read()
        return ret

    def read_sweep(this):
        return [*this["minmax"], this["step"]]

    def apply_condition(config, data):
        from ray import tune

        # TODO: tune.sample_from only supports random search algorithm.
        # To make conditional parameter for the other algorithms, different
        # algorithms should take different methods (will be added)
        if algorithm != "random":
            return config
        dp_pad_min = data["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"]["minmax"][0]
        # dp_pad_max = data['CELL_PAD_IN_SITES_DETAIL_PLACEMENT']['minmax'][1]
        dp_pad_step = data["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"]["step"]
        if dp_pad_step == 1:
            config["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"] = tune.sample_from(
                lambda spec: tune.randint(
                    dp_pad_min, spec.config.CELL_PAD_IN_SITES_GLOBAL_PLACEMENT + 1
                )
            )
        if dp_pad_step > 1:
            config["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"] = tune.sample_from(
                lambda spec: tune.choice(
                    np.ndarray.tolist(
                        np.arange(
                            dp_pad_min,
                            spec.config.CELL_PAD_IN_SITES_GLOBAL_PLACEMENT + 1,
                            dp_pad_step,
                        )
                    )
                )
            )
        return config

    def read_tune(this):
        from ray import tune

        min_, max_ = this["minmax"]
        if min_ == max_:
            # Returning a choice of a single element allow pbt algorithm to
            # work. pbt does not accept single values as tunable.
            return tune.choice([min_])
        if this["type"] == "int":
            if min_ == 0 and algorithm == "nevergrad":
                print(
                    "[WARNING TUN-0011] NevergradSearch may not work "
                    "with lower bound value 0."
                )
            if this["step"] == 1:
                return tune.randint(min_, max_)
            return tune.choice(np.ndarray.tolist(np.arange(min_, max_, this["step"])))
        if this["type"] == "float":
            if this["step"] == 0:
                return tune.uniform(min_, max_)
            return tune.choice(np.ndarray.tolist(np.arange(min_, max_, this["step"])))
        return None

    def read_tune_ax(name, this):
        from ray import tune

        dict_ = dict(name=name)
        min_, max_ = this["minmax"]
        if min_ == max_:
            dict_["type"] = "fixed"
            dict_["value"] = min_
        elif this["type"] == "int":
            if this["step"] == 1:
                dict_["type"] = "range"
                dict_["bounds"] = [min_, max_]
                dict_["value_type"] = "int"
            else:
                dict_["type"] = "choice"
                dict_["values"] = tune.randint(min_, max_, this["step"])
                dict_["value_type"] = "int"
        elif this["type"] == "float":
            if this["step"] == 1:
                dict_["type"] = "choice"
                dict_["values"] = tune.choice(
                    np.ndarray.tolist(np.arange(min_, max_, this["step"]))
                )
                dict_["value_type"] = "float"
            else:
                dict_["type"] = "range"
                dict_["bounds"] = [min_, max_]
                dict_["value_type"] = "float"
        return dict_

    def read_vizier(this):
        dict_ = {}
        min_, max_ = this["minmax"]
        dict_["value"] = (min_, max_)
        if "scale_type" in this:
            dict_["scale_type"] = this["scale_type"]
        if min_ == max_:
            dict_["type"] = "fixed"
        elif this["type"] == "int":
            dict_["type"] = "int"
        elif this["type"] == "float":
            dict_["type"] = "float"
        return dict_

    # Check file exists and whether it is a valid JSON file.
    assert os.path.isfile(file_name), f"File {file_name} not found."
    try:
        with open(file_name) as file:
            data = json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {file_name}")
    sdc_file = ""
    fr_file = ""
    if mode == "tune" and algorithm == "ax":
        config = list()
    else:
        config = dict()
    for key, value in data.items():
        if key == "best_result":
            continue
        if key == "_SDC_FILE_PATH" and value != "":
            if sdc_file != "":
                print("[WARNING TUN-0004] Overwriting SDC base file.")
            try:
                sdc_file = read(f"{os.path.dirname(file_name)}/{value}")
            except FileNotFoundError:
                sdc_file = SDC_TEMPLATE
            continue
        if key == "_FR_FILE_PATH" and value != "":
            if fr_file != "":
                print("[WARNING TUN-0005] Overwriting FastRoute base file.")
            fr_file = read(f"{os.path.dirname(file_name)}/{value}")
            continue
        if not isinstance(value, dict):
            config[key] = value
        elif mode == "sweep":
            config[key] = read_sweep(value)
        elif mode == "tune" and algorithm != "ax":
            config[key] = read_tune(value)
        elif mode == "tune" and algorithm == "ax":
            config.append(read_tune_ax(key, value))
        elif mode == "vizier":
            config[key] = read_vizier(value)
    if mode == "tune":
        config = apply_condition(config, data)
    return config, sdc_file, fr_file


def add_common_args(parser: argparse.ArgumentParser):
    # DUT
    parser.add_argument(
        "--design",
        type=str,
        metavar="<gcd,jpeg,ibex,aes,...>",
        required=True,
        help="Name of the design for Autotuning.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        metavar="<sky130hd,sky130hs,asap7,...>",
        required=True,
        help="Name of the platform for Autotuning.",
    )
    # Experiment Setup
    parser.add_argument(
        "--config",
        type=str,
        metavar="<path>",
        required=True,
        help="Configuration file that sets which knobs to use for Autotuning.",
    )
    parser.add_argument(
        "--to-stage",
        type=str,
        choices=("floorplan", "place", "cts", "route", "finish"),
        default=None,
        help="Run ORFS only to the given stage (inclusive)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="<float>",
        default=None,
        help="Time limit (in hours) for each trial run. Default is no limit.",
    )
    # Workload
    parser.add_argument(
        "--openroad_threads",
        type=int,
        metavar="<int>",
        default=16,
        help="Max number of threads openroad can use.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity level.\n\t0: only print status\n\t1: also print"
        " training stderr\n\t2: also print training stdout.",
    )
