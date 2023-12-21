//
// Created by lzzhao on 12/12/23.
//

#ifndef PYPSL_CUDA_PLANE_SWEEP_H
#define PYPSL_CUDA_PLANE_SWEEP_H

#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#include <filesystem>
#include <map>
#include <tuple>

#include <psl/exception.h>
#include <psl/cameraMatrix.h>
#include <psl/cudaPlaneSweep.h>

#include "json/json.hpp"


namespace pypsl {

namespace fs = std::filesystem;

class PlaneSweeper {
public:
    PlaneSweeper(const nlohmann::json& configs);

    void SetConfigs(const nlohmann::json& configs);
    void AddFrame(int frame_id, torch::Tensor image, const Eigen::Matrix4d& pose);
    void DeleteFrame(int frame_id);
    void ClearFrames();
    std::vector<int> GetFrameIds() const;
    torch::Tensor Process(int target_id);

private:
    void UpdateDistanceRange();
    int UploadImages(int target_id);

private:
    PSL::CudaPlaneSweep cPS_;
    Eigen::Matrix3d K_;
    std::map<int, std::tuple<cv::Mat, PSL::CameraMatrix<double>>> frame_map_;
};

}  // namespace pypsl

#endif //PYPSL_CUDA_PLANE_SWEEP_H
