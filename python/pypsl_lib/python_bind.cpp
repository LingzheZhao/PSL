//
// Created by lzzhao on 12/12/23.
//

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "json/json.hpp"
#include "pybind11/pybind11_eigen.h"
#include "pybind11/pybind11_filesystem.h"
#include "pybind11_json/pybind11_json.hpp"

#include "cps.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<pypsl::PlaneSweeper>(m, "PlaneSweeper")
        .def(pybind11::init<const nlohmann::json&>())
        .def("set_configs", &pypsl::PlaneSweeper::SetConfigs, "set configs by json")
        .def("add_frame", &pypsl::PlaneSweeper::AddFrame, "add frame")
        .def("delete_frame", &pypsl::PlaneSweeper::DeleteFrame, "delete frame")
        .def("clear_frames", &pypsl::PlaneSweeper::ClearFrames, "clear frames")
        .def("get_frame_ids", &pypsl::PlaneSweeper::GetFrameIds, "get frame ids")
        .def("process", &pypsl::PlaneSweeper::Process, "process")
        ;
}
