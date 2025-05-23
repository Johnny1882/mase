import logging
from typing import Tuple, Dict
import math
import os
import time
from multiprocessing import Process, Queue

import torch.fx as fx
from chop.passes.graph.utils import vf, v2p, init_project
import mase_components.helper.generate_memory as gen_lut
import torch.nn as nn

logger = logging.getLogger(__name__)

from .util import (
    get_verilog_parameters,
    add_node_name_to_keys_with_precision,
    get_top_param_name_with_precision,
    get_node_param_name_with_precision,
)
from pathlib import Path

from chop.ir.common import MASE_IMPLICIT_FUNCS

# =============================================================================
# Utilities
# =============================================================================


def _remove_last_comma(string):
    return string[0 : string.rfind(",")]


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def get_input_name(from_node, to_node):
    # Find name of to_node argument that comes from from_node

    if from_node == to_node:
        return "data_in_0"
    for key, val in to_node.meta["mase"].parameters["common"]["args"].items():
        if val["from"] == from_node:
            return key
    assert False, f"Cannot find edge from {from_node.name} to {to_node.name}"


def param_needs_signals(node, param, value, qualifier="data_in"):
    # Don't emit if it's function constant, but emit for any data_in or data_out
    # And any other parameters with storage interface specified as BRAM
    if type(value) != dict:
        # Constant function arguments don't have precision/shape info
        # TODO: change common metadata so constant arguments are passed with
        # is_constant flag
        return False
    if qualifier in param:
        return True
    else:
        return (
            node.meta["mase"].parameters["hardware"]["interface"][param]["storage"]
            == "BRAM"
        )


def is_real_input_arg(node, arg_idx):
    return (
        # Check if the argument index is within bounds
        arg_idx < len(node.args)
        # Ensure the argument is an instance of fx.Node
        and isinstance(node.args[arg_idx], fx.Node)
    ) and (
        # Check if the argument is in the implicit functions list
        node.args[arg_idx] in MASE_IMPLICIT_FUNCS
        or (
            # Check if the argument is an implicit function
            node.args[arg_idx].meta["mase"].parameters["common"]["mase_type"]
            == "implicit_func"
        )
        or (
            # Check if the argument is a placeholder input
            node.args[arg_idx].op
            == "placeholder"
        )
    )


# =============================================================================
# Verilog parameters
# =============================================================================


class VerilogParameterEmitter:
    def __init__(self, graph):
        self.graph = graph

    def emit(self, graph, parameter_map) -> Tuple[str, Dict[str, str]]:
        """
        Emit parameters at the top-level for the top-level module

        Returns Tuple:
        1) list of parameters as a string to be embedded in Verilog file
        """

        nodes_in = graph.nodes_in
        nodes_out = graph.nodes_out
        node_in_name = vf(nodes_in[0].name)
        node_out_name = vf(nodes_out[0].name)

        parameters = ""

        # Write node parameters
        for key, value in parameter_map.items():
            parameters += f"""    parameter {key} = {value},\n"""

        return _remove_last_comma(parameters)


# =============================================================================
# Verilog interface
# =============================================================================


class VerilogInterfaceEmitter:
    def __init__(self, graph):
        self.graph = graph

    def emit(self, graph, parameter_map):
        """
        Emit interface signal declarations for the top-level module
        """

        nodes_in = self.graph.nodes_in
        nodes_out = self.graph.nodes_out

        interface = ""
        # TODO: here we just enumerate the inputs of the input nodes - which may be
        # order insensitive and require manual connection when adding the graph to
        # a system.
        i = 0
        for node in nodes_in:
            # DiffLogic: input node is expecting binary array
            if "difflogic" in node.meta["mase"]["hardware"]["module"]:
                node_name = vf(node.name)
                interface += f"\n\tinput [({node_name}_DATA_IN_0_TENSOR_SIZE_DIM_0-1):0] data_in_0 [0:({node_name}_DATA_IN_0_TENSOR_SIZE_DIM_1-1)],"
                interface += f"\n\tinput  data_in_0_valid,"
                interface += f"\n\toutput data_in_0_ready,"
                continue

            node_name = vf(node.name)
            for arg_idx, arg in enumerate(
                node.meta["mase"].parameters["common"]["args"].keys()
            ):
                if is_real_input_arg(node, arg_idx):
                    # if "data_in" in arg:
                    arg_name = _cap(arg)
                    parallelism_params = [
                        param
                        for param in parameter_map
                        if param.startswith(f"{arg_name}_PARALLELISM_DIM")
                    ]
                    interface += f"""
    input  [{arg_name}_PRECISION_0-1:0] data_in_{i} [{'*'.join(parallelism_params)}-1:0],
    input  data_in_{i}_valid,
    output data_in_{i}_ready,"""
                    i += 1

        i = 0
        for node in nodes_out:
            # DiffLogic: output node is expecting binary array
            if "difflogic" in node.meta["mase"]["hardware"]["module"]:
                node_name = vf(node.name)
                node_result = "data_out_0"
                interface += f"\n\toutput [($clog2(({_cap(node_name)}_DATA_IN_0_TENSOR_SIZE_DIM_0/{_cap(node_name)}_{_cap(node_result)}_TENSOR_SIZE_DIM_0))):0] data_out_0 [0:({_cap(node_name)}_{_cap(node_result)}_TENSOR_SIZE_DIM_0-1)],"
                interface += f"\n\toutput  data_out_0_valid,"
                interface += f"\n\tinput data_out_0_ready,"
                continue

            node_name = vf(node.name)
            for result in node.meta["mase"].parameters["common"]["results"].keys():
                if "data_out" in result:
                    result_name = _cap(result)
                    parallelism_params = [
                        param
                        for param in parameter_map
                        if param.startswith(f"{result_name}_PARALLELISM_DIM")
                    ]
                    interface += f"""
    output  [{result_name}_PRECISION_0-1:0] data_out_{i} [{'*'.join(parallelism_params)}-1:0],
    output  data_out_{i}_valid,
    input data_out_{i}_ready,"""
                    i += 1

        # TODO: emit off-chip parameter interface

        return _remove_last_comma(interface)


# =============================================================================
# Verilog signals
# =============================================================================


class VerilogSignalEmitter:
    def __init__(self, graph):
        self.graph = graph

    def _emit_signals_top_internal(self, node, parameter_map):
        signals = ""
        node_name = vf(node.name)
        # Input signals
        for arg, arg_info in node.meta["mase"].parameters["common"]["args"].items():
            if not isinstance(arg_info, dict):
                continue

            # DiffLogic: signals depends on different things
            if node.meta["mase"]["hardware"].get("module") in [
                "fixed_difflogic_logic",
                "fixed_difflogic_groupsum",
            ]:
                signals += f"\nlogic [({_cap(node_name)}_{_cap(arg)}_TENSOR_SIZE_DIM_0-1):0] {node_name}_{arg};"
                signals += f"\nlogic {node_name}_{arg}_valid;"
                signals += f"\nlogic {node_name}_{arg}_ready;"
                continue
            elif node.meta["mase"]["hardware"].get("module") in [
                "fixed_difflogic_flatten"
            ]:
                signals += f"\nlogic [({_cap(node_name)}_{_cap(arg)}_TENSOR_SIZE_DIM_0-1):0] {node_name}_{arg} [0:({_cap(node_name)}_{_cap(arg)}_TENSOR_SIZE_DIM_1-1)];"
                signals += f"\nlogic {node_name}_{arg}_valid;"
                signals += f"\nlogic {node_name}_{arg}_ready;"
                continue

            # Skip off-chip parameters as they will be directly connected to the top level
            if (
                "data_in" in arg
                or node.meta["mase"].parameters["hardware"]["interface"][arg]["storage"]
                == "BRAM"
            ):
                arg_name = v2p(arg)
                parallelism_params = [
                    param
                    for param in parameter_map
                    if f"{node_name}_{arg_name}_PARALLELISM_DIM" in param
                ]

                # Getitem argument always get mapped to port 0 irrespective of
                # actual argument index
                if node.meta["mase"]["common"]["mase_op"] == "getitem":
                    arg = "data_in_0"

                signals += f"""
logic [{node_name}_{arg_name}_PRECISION_0-1:0]  {node_name}_{arg}        [{'*'.join(parallelism_params)}-1:0];
logic                             {node_name}_{arg}_valid;
logic                             {node_name}_{arg}_ready;"""

        # Output signals
        for result, result_info in (
            node.meta["mase"].parameters["common"]["results"].items()
        ):
            if not isinstance(result_info, dict):
                continue

            # DiffLogic: signals depends on different things
            if node.meta["mase"]["hardware"].get("module") in ["fixed_difflogic_logic"]:
                signals += f"\nlogic [({_cap(node_name)}_{_cap(result)}_TENSOR_SIZE_DIM_0-1):0] {node_name}_{result};"
                signals += f"\nlogic {node_name}_{result}_valid;"
                signals += f"\nlogic {node_name}_{result}_ready;"
                continue
            elif node.meta["mase"]["hardware"].get("module") in [
                "fixed_difflogic_groupsum"
            ]:
                signals += f"\nlogic [($clog2(({_cap(node_name)}_DATA_IN_0_TENSOR_SIZE_DIM_0/{_cap(node_name)}_{_cap(result)}_TENSOR_SIZE_DIM_0))):0] {node_name}_{result} [0:({_cap(node_name)}_{_cap(result)}_TENSOR_SIZE_DIM_0-1)];"
                signals += f"\nlogic {node_name}_{result}_valid;"
                signals += f"\nlogic {node_name}_{result}_ready;"
                continue
            elif node.meta["mase"]["hardware"].get("module") in [
                "fixed_difflogic_flatten"
            ]:
                signals += f"\nlogic [({_cap(node_name)}_{_cap(result)}_TENSOR_SIZE_DIM_0-1):0] {node_name}_{result};"
                signals += f"\nlogic {node_name}_{result}_valid;"
                signals += f"\nlogic {node_name}_{result}_ready;"
                continue

            # Skip off-chip parameters as they will be directly connected to the top level
            if (
                "data_out" in result
                or node.meta["mase"].parameters["hardware"]["interface"][result][
                    "storage"
                ]
                == "BRAM"
            ):
                result_name = v2p(result)
                parallelism_params = [
                    param
                    for param in parameter_map
                    if f"{node_name}_{result_name}_PARALLELISM_DIM" in param
                ]
                signals += f"""
logic [{node_name}_{result_name}_PRECISION_0-1:0]  {node_name}_{result}        [{'*'.join(parallelism_params)}-1:0];
logic                             {node_name}_{result}_valid;
logic                             {node_name}_{result}_ready;"""

        return signals

    def _emit_signals_top_hls(self, node, parameter_map):
        """
        TODO
        """

        node_name = vf(node.name)
        # Control signals for HLS component
        signals = f"""
logic {node_name}_start;
logic {node_name}_done;
logic {node_name}_idle;
logic {node_name}_ready;
logic {node_name}_ce;"""

        # Input signals
        for key, value in node.meta["mase"].parameters["common"]["args"].items():
            # No internal signals if the memory is stored off chip
            if not param_needs_signals(node, key, value, qualifier="data_in"):
                continue

            cap_key = v2p(key)
            size = math.prod(value["shape"])

            if key != "data_in":
                a_width = math.ceil(math.log2(size))
            else:
                depth = parameter_map[f"{node_name}_{cap_key}_DEPTH"]
                a_width = math.ceil(math.log2(depth * size))

            signals += f"""
logic [{node_name}_{cap_key}_PRECISION_0-1:0]  {node_name}_{key}_q0;
logic [{a_width}-1:0]                    {node_name}_{key}_address0;
logic                                    {node_name}_{key}_ce0;"""

        # Output signals
        for key, value in node.meta["mase"].parameters["common"]["results"].items():
            # No internal signals if the memory is stored off chip
            if not param_needs_signals(node, key, value, qualifier="data_out"):
                continue

            cap_key = v2p(key)
            size = math.prod(value["shape"])
            a_width = math.ceil(math.log2(size))
            signals += f"""
logic [{node_name}_{cap_key}_PRECISION_0-1:0]  {node_name}_{key}_d0;
logic [{a_width}-1:0]                    {node_name}_{key}_address0;
logic                                    {node_name}_{key}_ce0;
logic                                    {node_name}_{key}_we0;"""
        return signals

    def emit(self, graph, parameter_map):
        """
        Emit internal signal declarations for the top-level module
        """

        signals = ""
        for node in graph.fx_graph.nodes:
            if node.meta["mase"].parameters["hardware"]["is_implicit"]:
                continue
            node_name = vf(node.name)
            signals += f"""
// --------------------------
//   {node_name} signals
// --------------------------"""
            if "INTERNAL" in node.meta["mase"].parameters["hardware"]["toolchain"]:
                signals += self._emit_signals_top_internal(node, parameter_map)
            elif node.meta["mase"].parameters["hardware"]["toolchain"] == "HLS":
                signals += self._emit_signals_top_hls(node, parameter_map)
            else:
                assert False, "Unknown node toolchain for signal declarations."

        return signals


# =============================================================================
# Verilog components (INTERNAL)
# =============================================================================


class VerilogInternalComponentEmitter:
    def __init__(self, graph):
        self.graph = graph

    def _emit_module_parameters_top_internal(self, key, value, node, parameter_map):
        node_name = vf(node.name)
        component_name = f"{node_name}_{key}_source"
        component_name_inst = f"{component_name}_0"

        parameters = ""
        for param in node.meta["mase"].parameters["hardware"]["verilog_param"].keys():
            if f"{_cap(key)}_" in param:
                parameters += f"    .{param}({node_name}_{param}),\n"
        parameters = _remove_last_comma(parameters)

        return f"""
{component_name} #(
{parameters}
) {component_name_inst} (
    .clk(clk),
    .rst(rst),
    .data_out({node_name}_{key}),
    .data_out_ready({node_name}_{key}_ready),
    .data_out_valid({node_name}_{key}_valid)
);
"""

    def _emit_getitem_signals(self, node):
        """
        Getitem nodes have arg list like (None, None, None, Arg, None, None)
        where the meaningful arg is at an arbitrary index, but always maps to
        data_in_0 interface of the hardware
        """

        node_name = vf(node.name)

        return f"""
    .data_in_0       ({node_name}_data_in_0),
    .data_in_0_valid ({node_name}_data_in_0_valid),
    .data_in_0_ready ({node_name}_data_in_0_ready),
    
    .data_out_0       ({node_name}_data_out_0),
    .data_out_0_valid ({node_name}_data_out_0_valid),
    .data_out_0_ready ({node_name}_data_out_0_ready),
        """

    def emit(self, node, parameter_map):
        node_name = vf(node.name)
        component_name = node.meta["mase"].parameters["hardware"]["module"]
        signals = ""

        # Emit component instantiation parameters
        parameters = ""
        for key, value in (
            node.meta["mase"].parameters["hardware"]["verilog_param"].items()
        ):
            if value is None:
                continue
            if "[" in key:
                dict_key = add_node_name_to_keys_with_precision(node_name, key)
                top_param_key = get_top_param_name_with_precision(dict_key)
                node_param_key = get_node_param_name_with_precision(node_name, dict_key)
            else:
                dict_key = f"{node_name}_{key}"
                top_param_key = f"{node_name}_{key}"
                node_param_key = key

            key_value = parameter_map[dict_key]
            debug_info = f"// = {key_value}"
            parameters += f"""    .{node_param_key}({top_param_key}),\n"""  # hack: debug info removed for now as debug info may contain commas

        parameters = _remove_last_comma(parameters)

        # Handle getitem nodes separately since an arbitrary argument index
        # will always be mapped to data_in_0 interface of the hardware
        if node.meta["mase"]["common"]["mase_op"] == "getitem":
            signals += self._emit_getitem_signals(node)

        # All other node types
        else:
            # Emit component instantiation input signals
            for key, value in node.meta["mase"].parameters["common"]["args"].items():
                if "inplace" in key or not isinstance(value, dict):
                    continue

                # DiffLogic: do not emit storage interface
                # if node.meta["mase"]["hardware"].get("module") in ["fixed_difflogic_logic", "fixed_difflogic_logic"] and ("data_in" not in key):
                #     continue

                signals += f"""
    .{key}({node_name}_{key}),
    .{key}_valid({node_name}_{key}_valid),
    .{key}_ready({node_name}_{key}_ready),
        """

            # Emit component instantiation output signals
            for key, value in node.meta["mase"].parameters["common"]["results"].items():
                signals += f"""
    .{key}({node_name}_{key}),
    .{key}_valid({node_name}_{key}_valid),
    .{key}_ready({node_name}_{key}_ready),
        """

        # Remove final comma in signal list
        signals = _remove_last_comma(signals)

        # Combine component instantiation
        components = f"""
// {node_name}
{component_name} #(
{parameters}
) {node_name}_inst (
    .clk(clk),
    .rst(rst),
{signals}
);
"""

        # DiffLogic: do not need to generate weights and biases
        # if node.meta["mase"]["hardware"].get("module") in ["fixed_difflogic_logic", "fixed_difflogic_logic"]:
        #     return components

        # Emit module parameter instances (e.g. weights and biases)
        for arg, arg_info in node.meta["mase"].parameters["common"]["args"].items():
            if "data_in" in arg:
                continue
            if not isinstance(arg_info, dict):
                continue

            components += self._emit_module_parameters_top_internal(
                arg, arg_info, node, parameter_map
            )

        return components


# =============================================================================
# Verilog components (HLS)
# =============================================================================


class VerilogHLSComponentEmitter:
    def __init__(self, graph):
        self.graph = graph

    def _emit_module_parameters_top_hls(self, key, value, node, parameter_map):
        node_name = vf(node.name)
        cap_key = v2p(key)
        component_name = f"{node_name}_{key}_source"
        component_name_inst = f"{node_name}_{key}_0"

        size_debug_info = math.prod(value["shape"])
        a_width = math.ceil(math.log2(size_debug_info))

        return f"""
{component_name} #(
    .DATA_WIDTH({node_name}_{cap_key}_PRECISION_0),
    .ADDR_RANGE({node_name}_{cap_key}_TENSOR_SIZE_0),
    .ADDR_WIDTH({a_width})
) {component_name_inst} (
    .clk(clk),
    .reset(rst),

    .address0({node_name}_{key}_address0),
    .ce0({node_name}_{key}_ce0),
    .q0({node_name}_{key}_q0)
);
"""

    def emit(self, node, parameter_map):
        node_name = vf(node.name)
        component_name = node.meta["mase"].parameters["hardware"]["module"]

        # Emit kernel instance
        signals = ""
        for key, value in node.meta["mase"].parameters["common"]["args"].items():
            signals += f"""
    .{key}_address0({node_name}_{key}_address0),
    .{key}_ce0({node_name}_{key}_ce0),
    .{key}_q0({node_name}_{key}_q0),
"""

        for key, value in node.meta["mase"].parameters["common"]["results"].items():
            signals += f"""
    .{key}_address0({node_name}_{key}_address0),
    .{key}_ce0({node_name}_{key}_ce0),
    .{key}_we0({node_name}_{key}_we0),
    .{key}_d0({node_name}_{key}_d0),
"""
        signals = _remove_last_comma(signals)
        components = f"""
// {node_name}
{component_name} #(
) {node_name}_inst (
    .ap_clk(clk),
    .ap_rst(rst),

    .ap_start({node_name}_start),
    .ap_idle({node_name}_idle),
    .ap_ready({node_name}_ready),
    .ap_done({node_name}_done),
    .ap_ce({node_name}_ce),
{signals}
);
"""

        # Emit parameter instance
        for key, value in node.meta["mase"].parameters["common"]["args"].items():
            # Skip the parameter instance if the memory is stored off chip
            if not param_needs_signals(node, key, value, qualifier="data_in"):
                continue
            components += self._emit_module_parameters_top_hls(
                key, value, node, parameter_map
            )

        for key, value in node.meta["mase"].parameters["common"]["results"].items():
            # Skip the parameter instance if the memory is stored off chip
            if not param_needs_signals(node, key, value, qualifier="data_out"):
                continue
            components += self._emit_module_parameters_top_hls(
                key, value, node, parameter_map
            )

        return components


# =============================================================================
# Verilog components
# =============================================================================


class VerilogComponentEmitter:
    def __init__(self, graph):
        self.graph = graph
        self.internal_emitter = VerilogInternalComponentEmitter(graph)
        self.hls_emitter = VerilogHLSComponentEmitter(graph)

    def emit(self, graph, parameter_map):
        """
        Emit component declarations for the top-level module
        """

        components = """
// --------------------------
//   Component instantiation
// --------------------------
"""
        for node in graph.fx_graph.nodes:
            if node.meta["mase"].parameters["hardware"]["is_implicit"]:
                continue
            if "INTERNAL" in node.meta["mase"].parameters["hardware"]["toolchain"]:
                components += self.internal_emitter.emit(node, parameter_map)
            elif node.meta["mase"].parameters["hardware"]["toolchain"] == "HLS":
                components += self.hls_emitter.emit(node, parameter_map)
            else:
                assert False, "Unknown node toolchain for signal declarations."

        return components


# =============================================================================
# Verilog wires
# =============================================================================


class VerilogWireEmitter:
    def __init__(self, graph, parameter_map):
        self.graph = graph
        self.parameter_map = parameter_map

        self.wires = """
// --------------------------
//   Interconnections
// --------------------------
    """

    def _emit_top_wires(self):
        nodes_in = self.graph.nodes_in
        nodes_out = self.graph.nodes_out

        # ============================================================
        # Top level wires
        # ============================================================

        wires = ""
        # TODO: here we just enumerate the inputs of the input nodes - which may be
        # order insensitive and require manual connection when adding the graph to
        # a system.
        i = 0
        for node in nodes_in:
            node_name = vf(node.name)
            for arg_idx, arg in enumerate(
                node.meta["mase"].parameters["common"]["args"].keys()
            ):
                if is_real_input_arg(node, arg_idx):
                    wires += f"""
assign data_in_{i}_ready = {node_name}_{arg}_ready;
assign {node_name}_{arg}_valid    = data_in_{i}_valid;
assign {node_name}_{arg}    = data_in_{i};
"""
                    i += 1
        i = 0
        for node in nodes_out:
            node_name = vf(node.name)
            for result in node.meta["mase"].parameters["common"]["results"].keys():
                if "data_out" in result:
                    wires += f"""
assign data_out_{i}_valid = {node_name}_{result}_valid;
assign {node_name}_{result}_ready    = data_out_{i}_ready;
assign data_out_{i} = {node_name}_{result};
"""
                    i += 1

        # TODO: emit off-chip parameter interface

        return wires

    def _emit_getitem_wires(self, node):
        """
        Getitem nodes may receive an output from an arbitrary index of the parent node,
        which is always driven to port 0 of the getitem node
        """

        from_name = vf(node.args[0].name)
        to_name = vf(node.name)
        select = node.args[1]

        return f"""
assign {from_name}_data_out_{select}_ready  = {to_name}_data_in_0_ready;
assign {to_name}_data_in_0_valid    = {from_name}_data_out_{select}_valid;
assign {to_name}_data_in_0 = {from_name}_data_out_{select};
"""

    def _emit_node2node_wires(self):
        nodes_in = self.graph.nodes_in

        wires = ""
        for node in self.graph.fx_graph.nodes:

            if (
                # Skip implicit nodes
                node.meta["mase"].parameters["hardware"]["is_implicit"]
                # Input nodes were already connected by the previous process
                or node in nodes_in
            ):
                continue

            # Getitem nodes are handled separately
            if node.meta["mase"]["common"]["mase_op"] == "getitem":
                wires += self._emit_getitem_wires(node)
                continue

            to_name = vf(node.name)

            for i, node_in in enumerate(node.all_input_nodes):
                from_name = vf(node_in.name)
                wires += f"""
assign {from_name}_data_out_0_ready  = {to_name}_data_in_{i}_ready;
assign {to_name}_data_in_{i}_valid    = {from_name}_data_out_0_valid;
assign {to_name}_data_in_{i} = {from_name}_data_out_0;
"""
        return wires

    def emit(self):
        """
        Emit internal signal connections for the top-level module
        This includes two interconnection types:
        1. Type casting between inputs and outputs
        2. Interface casting between inputs and outputs
        """

        self.wires += self._emit_top_wires()
        self.wires += self._emit_node2node_wires()
        return self.wires


# =============================================================================
# Emit Verilog
# =============================================================================


class VerilogEmitter:
    def __init__(self, graph):
        self.graph = graph

        self.parameter_map = get_verilog_parameters(graph)

    def emit(self, graph, top_name):
        parameters_to_emit = VerilogParameterEmitter(graph).emit(
            graph, self.parameter_map
        )

        interface_to_emit = VerilogInterfaceEmitter(graph).emit(
            graph, self.parameter_map
        )

        signals_to_emit = VerilogSignalEmitter(graph).emit(graph, self.parameter_map)

        components_to_emit = VerilogComponentEmitter(graph).emit(
            graph, self.parameter_map
        )

        wires_to_emit = VerilogWireEmitter(graph, self.parameter_map).emit()

        time_to_emit = time.strftime("%d/%m/%Y %H:%M:%S")

        module_inst = """
// =====================================
//     Mase Hardware
//     Model: {}
//     {}
// =====================================
`timescale 1ns/1ps
module {} #(
{}
) (
    input clk,
    input rst,
{}
);
{}
{}
{}
endmodule
    """.format(
            top_name,
            time_to_emit,
            top_name,
            parameters_to_emit,
            interface_to_emit,
            signals_to_emit,
            components_to_emit,
            wires_to_emit,
        )
        return module_inst


def emit_verilog_top_transform_pass(graph, pass_args={}):
    """Emit the top-level model design in Verilog

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)


    - pass_args
        - project_dir -> str : the directory of the project for cosimulation
        - top_name -> str : top-level name
    """

    logger.info("Emitting Verilog...")

    # Create project directory, and the verilog is emmited to {project_name}/hardware/rtl
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"
    init_project(project_dir)
    rtl_dir = os.path.join(project_dir, "hardware", "rtl")

    top = VerilogEmitter(graph).emit(graph, top_name)

    top_file = os.path.join(rtl_dir, f"{top_name}.sv")
    with open(top_file, "w") as top_design:
        top_design.write(top)

    # Code to generate the LUTs for activation functions. Currently here because RTL dir is required.
    # Move to verilog emitter if you can pass path somehow.
    # Alternatively, add a class to the emitter that can be called to generate LUTs, for LUT based implementations of activation functions,
    # or other functions that require LUTs such as PolyLUT or LUTnet neurons.
    for node in graph.fx_graph.nodes:
        # print(vars(node))
        # print(type(node))
        if node.op == "call_module":
            module = dict(graph.model.named_modules())[node.target]
            if isinstance(module, nn.SiLU):
                func = "silu"
            elif isinstance(module, nn.ELU):
                func = "elu"
            elif isinstance(module, nn.Sigmoid):
                func = "sigmoid"
            elif isinstance(module, nn.LogSigmoid):
                func = "logsigmoid"
            elif isinstance(module, nn.Softmax):
                func = "exp"
            else:
                func = "Unknown"

            if func != "Unknown":
                d_in_width = node.meta["mase"].parameters["hardware"]["verilog_param"][
                    "DATA_IN_0_PRECISION_0"
                ]
                d_in_f_width = node.meta["mase"].parameters["hardware"][
                    "verilog_param"
                ]["DATA_IN_0_PRECISION_1"]
                d_out_width = node.meta["mase"].parameters["hardware"]["verilog_param"][
                    "DATA_OUT_0_PRECISION_0"
                ]
                d_out_f_width = node.meta["mase"].parameters["hardware"][
                    "verilog_param"
                ]["DATA_OUT_0_PRECISION_1"]
                gen_lut.generate_sv_lut(
                    func,
                    d_in_width,
                    d_in_f_width,
                    d_out_width,
                    d_out_f_width,
                    path=rtl_dir,
                    path_with_dtype=False,
                )
    return graph, {}
