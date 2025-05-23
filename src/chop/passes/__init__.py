from . import graph
from . import module
from . import onnx

from .graph.analysis import (
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
    profile_statistics_analysis_pass,
    report_graph_analysis_pass,
    report_node_hardware_type_analysis_pass,
    report_node_meta_param_analysis_pass,
    report_node_shape_analysis_pass,
    report_node_type_analysis_pass,
    verify_common_metadata_analysis_pass,
    run_cosim_analysis_pass,
    get_synthesis_results,
    add_movement_metadata_analysis_pass,
)
from .graph.transforms import (
    prune_transform_pass,
    prune_detach_hook_transform_pass,
    # prune_unwrap_transform_pass,
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
    conv_bn_fusion_transform_pass,
    logicnets_fusion_transform_pass,
    onnx_annotate_transform_pass,
    partition_to_multi_device_transform_pass,
    emit_verilog_top_transform_pass,
    emit_bram_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_cocotb_transform_pass,
    emit_vivado_project_transform_pass,
    raise_granularity_transform_pass,
    patch_metadata_transform_pass,
    insert_lora_adapter_transform_pass,
    fuse_lora_weights_transform_pass,
    ann2snn_transform_pass,
)
from .module.analysis import calculate_avg_bits_module_analysis_pass
from .module.transforms import (
    quantize_module_transform_pass,
    resharding_transform_pass,
    ann2snn_module_transform_pass,
)

from .onnx.analysis import (
    export_fx_graph_analysis_pass,
)

from .graph.analysis.autosharding import autosharding_analysis_pass
