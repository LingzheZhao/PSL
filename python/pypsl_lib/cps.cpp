//
// Created by lzzhao on 12/12/23.
//

#include <opencv2/opencv.hpp>

#include "cps.h"

namespace pypsl {

PlaneSweeper::PlaneSweeper(const nlohmann::json &configs) {
    this->SetConfigs(configs);
}

void PlaneSweeper::AddFrame(int frame_id, torch::Tensor image, const Eigen::Matrix4d& pose) {
    CHECK_CONTIGUOUS(image);
    int H = image.sizes()[0];
    int W = image.sizes()[1];
    int C = image.sizes()[2];
    assert (C == 3 || C == 1);
    Eigen::Matrix3d R = pose.block<3,3>(0,0);
    Eigen::Vector3d t = pose.block<3,1>(0,3);
    frame_map_[frame_id] = std::make_tuple<cv::Mat, PSL::CameraMatrix<double>>(
        cv::Mat(H, W, CV_8UC3, image.data_ptr<unsigned char>()),
        PSL::CameraMatrix<double>()
    );
    std::get<1>(frame_map_[frame_id]).setKRT(K_, R, t);
}

void PlaneSweeper::DeleteFrame(int frame_id) {
    frame_map_.erase(frame_id);
}

void PlaneSweeper::ClearFrames() {
    frame_map_.clear();
}

std::vector<int> PlaneSweeper::GetFrameIds() const {
    std::vector<int> frame_ids;
    for (const auto& [frame_id, frame] : frame_map_) {
        frame_ids.push_back(frame_id);
    }
    return frame_ids;
}

torch::Tensor PlaneSweeper::Process(int target_id) {
    this->UpdateDistanceRange();

    // now we upload the images
    int ref_id = this->UploadImages(target_id);

    // now we run the plane sweep
    cPS_.process(ref_id);
    std::cout << "Plane sweep done" << std::endl;

    cv::Mat ref_image = cPS_.downloadImage(ref_id);
    cv::imshow("ref", ref_image);
    cv::waitKey(0);
    PSL::DepthMap<float, double> depth_map = cPS_.getBestDepth();
    depth_map.displayInvDepthColored(0.109379, 4.37517, 100);
//    cv::Mat depth_image(depth_map.getHeight(), depth_map.getWidth(), CV_32FC1, depth_map.getDataPtr());
//    cv::imshow("depth", depth_image);
//    cv::waitKey(0);
    auto depth_tensor = torch::from_blob(
//        depth_image.data,
//        {depth_image.rows, depth_image.cols},
        depth_map.getDataPtr(),
        {depth_map.getHeight(), depth_map.getWidth()},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    );
    return depth_tensor.clone();
}

int PlaneSweeper::UploadImages(int target_id) {
    int ref_id = -1;
//    for (auto& [frame_id, frame] : frame_map_) {
//        int id = cPS_.addImage(std::get<0>(frame), std::get<1>(frame));
//        if (frame_id == target_id) {
//            ref_id = id;
//        }
//    }
    // add first 5 images
    for (int i = 0; i < 5; i++) {
        int id = cPS_.addImage(std::get<0>(frame_map_[i]), std::get<1>(frame_map_[i]));
        if (i == target_id) {
            ref_id = id;
        }
    }
    if (ref_id == -1) {
        PSL_THROW_EXCEPTION("Target id not found");
    }
    return ref_id;
}

void PlaneSweeper::UpdateDistanceRange() {
    // Each of the datasets contains 25 cameras taken in 5 rows
    // The reconstructions are not metric. In order to have an idea about the scale
    // everything is defined with respect to the average distance between the cameras.
    double avgDistance = 0;
    int numDistances = 0;

    // calculate average distance between cameras
    for (auto& [frame_id, frame] : frame_map_) {
        for (auto& [frame_id2, frame2] : frame_map_) {
            if (frame_id < frame_id2) {
                avgDistance += (std::get<1>(frame).getC() - std::get<1>(frame2).getC()).norm();
                numDistances++;
            }
        }
    }
    if (numDistances < 2)
    {
        PSL_THROW_EXCEPTION("Could not compute average distance, less than two cameras found");
    }
    avgDistance /= numDistances;
    std::cout << "numDistances: " << numDistances << std::endl;
    std::cout << "Cameras have an average distance of " << avgDistance << "." << std::endl;

    double minZ = 2.5 * avgDistance;
    double maxZ = 100.0 * avgDistance;
    std::cout << "  Z range :  " << minZ << "  - " << maxZ <<  std::endl;
    cPS_.setZRange(minZ, maxZ);
}

void PlaneSweeper::SetConfigs(const nlohmann::json &configs) {
    // Camera matrix
    if (configs.contains("pinhole_intrinsics")) {
        auto K = configs["pinhole_intrinsics"];
        if (K.contains("fx") && K.contains("fy") && K.contains("cx") && K.contains("cy")) {
            assert(K["fx"].is_number_float());
            assert(K["fy"].is_number_float());
            assert(K["cx"].is_number_float());
            assert(K["cy"].is_number_float());
            K_(0,0) = K["fx"];
            K_(1,1) = K["fy"];
            K_(0,2) = K["cx"];
            K_(1,2) = K["cy"];
        } else {
            PSL_THROW_EXCEPTION("camera_intrinsics must contain fx, fy, cx, cy");
        }
    } else {
        PSL_THROW_EXCEPTION("configs must contain camera_intrinsics");
    }

    // Scale
    if (configs.contains("scale")) {
        assert(configs["scale"].is_number_float());
        cPS_.setScale(configs["scale"]);
    } else {
        // default: 0.25
        cPS_.setScale(0.25);
    }

    // Match window size
    if (configs.contains("match_window_size")) {
        auto match_window_size = configs["match_window_size"];
        if (match_window_size.contains("width") && match_window_size.contains("height")) {
            assert(match_window_size["width"].is_number_integer());
            assert(match_window_size["height"].is_number_integer());
            cPS_.setMatchWindowSize(match_window_size["width"], match_window_size["height"]);
        } else {
            PSL_THROW_EXCEPTION("match_window_size must contain width and height");
        }
    } else {
        // default: 7, 7
        cPS_.setMatchWindowSize(7, 7);
    }

    // Num planes
    if (configs.contains("num_planes")) {
        assert(configs["num_planes"].is_number_integer());
        cPS_.setNumPlanes(configs["num_planes"]);
    } else {
        // default: 256
        cPS_.setNumPlanes(256);
    }

    // Occlusion mode: NONE, REF_SPLIT, BEST_K
    if (configs.contains("occlusion_mode")) {
        std::string occlusion_mode = configs["occlusion_mode"];
        if (occlusion_mode == "NONE") {
            cPS_.setOcclusionMode(PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_NONE);
        } else if (occlusion_mode == "REF_SPLIT") {
            cPS_.setOcclusionMode(PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_REF_SPLIT);
        } else if (occlusion_mode == "BEST_K") {
            if (configs.contains("best_k")) {
                assert(configs["best_k"].is_number_integer());
                cPS_.setOcclusionBestK(configs["best_k"]);
            } else {
                PSL_THROW_EXCEPTION("occlusion_mode is BEST_K, but best_k is not set");
            }
            cPS_.setOcclusionMode(PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_BEST_K);
        } else {
            PSL_THROW_EXCEPTION("occlusion_mode must be NONE, REF_SPLIT, or BEST_K");
        }
    } else {
        // default: REF_SPLIT
        cPS_.setOcclusionMode(PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_REF_SPLIT);
    }

    // Plane generation mode: UNIFORM_DEPTH, UNIFORM_DISPARITY
    if (configs.contains("plane_generation_mode")) {
        std::string plane_generation_mode = configs["plane_generation_mode"];
        if (plane_generation_mode == "UNIFORM_DEPTH") {
            cPS_.setPlaneGenerationMode(PSL::PlaneSweepPlaneGenerationMode::PLANE_SWEEP_PLANEMODE_UNIFORM_DEPTH);
        } else if (plane_generation_mode == "UNIFORM_DISPARITY") {
            cPS_.setPlaneGenerationMode(PSL::PlaneSweepPlaneGenerationMode::PLANE_SWEEP_PLANEMODE_UNIFORM_DISPARITY);
        } else {
            PSL_THROW_EXCEPTION("plane_generation_mode must be UNIFORM_DEPTH or UNIFORM_DISPARITY");
        }
    } else {
        // default: UNIFORM_DISPARITY
        cPS_.setPlaneGenerationMode(PSL::PlaneSweepPlaneGenerationMode::PLANE_SWEEP_PLANEMODE_UNIFORM_DISPARITY);
    }

    // Matching costs: SAD, ZNCC
    if (configs.contains("matching_costs")) {
        std::string matching_costs = configs["matching_costs"];
        if (matching_costs == "SAD") {
            cPS_.setMatchingCosts(PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD);
        } else if (matching_costs == "ZNCC") {
            cPS_.setMatchingCosts(PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_ZNCC);
        } else {
            PSL_THROW_EXCEPTION("matching_costs must be SAD or ZNCC");
        }
    } else {
        // default: SAD
        cPS_.setMatchingCosts(PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD);
    }


    // Sub pixel interpolation mode: DIRECT, INVERSE
    if (configs.contains("sub_pixel_interpolation_mode")) {
        std::string sub_pixel_interpolation_mode = configs["sub_pixel_interpolation_mode"];
        if (sub_pixel_interpolation_mode == "DIRECT") {
            cPS_.setSubPixelInterpolationMode(PSL::PlaneSweepSubPixelInterpMode::PLANE_SWEEP_SUB_PIXEL_INTERP_DIRECT);
        } else if (sub_pixel_interpolation_mode == "INVERSE") {
            cPS_.setSubPixelInterpolationMode(PSL::PlaneSweepSubPixelInterpMode::PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
        } else {
            PSL_THROW_EXCEPTION("sub_pixel_interpolation_mode must be DIRECT or INVERSE");
        }
    } else {
        // default: INVERSE
        cPS_.setSubPixelInterpolationMode(PSL::PlaneSweepSubPixelInterpMode::PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
    }

    // Enable color matching
    if (configs.contains("enable_color_matching")) {
        assert (configs["enable_color_matching"].is_boolean());
        cPS_.enableColorMatching(configs["enable_color_matching"]);
    } else {
        // default: true
        cPS_.enableColorMatching(true);
    }

    // Enable output best costs
    if (configs.contains("enable_output_best_costs")) {
        assert (configs["enable_output_best_costs"].is_boolean());
        cPS_.enableOutputBestCosts(configs["enable_output_best_costs"]);
    } else {
        // default: false
        cPS_.enableOutputBestCosts(false);
    }

    // Enable output best depth
    if (configs.contains("enable_output_best_depth")) {
        assert (configs["enable_output_best_depth"].is_boolean());
        cPS_.enableOutputBestDepth(configs["enable_output_best_depth"]);
    } else {
        // default: true
        cPS_.enableOutputBestDepth(true);
    }

    // Enable output cost volume
    if (configs.contains("enable_output_cost_volume")) {
        assert (configs["enable_output_cost_volume"].is_boolean());
        cPS_.enableOutputCostVolume(configs["enable_output_cost_volume"]);
    } else {
        // default: false
        cPS_.enableOutputCostVolume(false);
    }

    // Enable output uniqueness ratio
    if (configs.contains("enable_output_uniqueness_ratio")) {
        assert (configs["enable_output_uniqueness_ratio"].is_boolean());
        cPS_.enableOutputUniquenessRatio(configs["enable_output_uniqueness_ratio"]);
    } else {
        // default: false
        cPS_.enableOutputUniquenessRatio(false);
    }

    // Enable sub pixel
    if (configs.contains("enable_sub_pixel")) {
        assert (configs["enable_sub_pixel"].is_boolean());
        cPS_.enableSubPixel(configs["enable_sub_pixel"]);
    } else {
        // default: true
        cPS_.enableSubPixel(true);
    }
}

} // namespace pypsl
