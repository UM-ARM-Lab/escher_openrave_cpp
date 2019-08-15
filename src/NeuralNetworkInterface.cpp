
#include "Utilities.hpp"

float ClassificationModel::predict(Eigen::VectorXd input)
{
    Eigen::VectorXd normalized_input = (input - input_mean_).cwiseQuotient(input_std_);

    fdeep::float_vec normalized_input_vec(normalized_input.data(), normalized_input.data() + normalized_input.size());
    fdeep::shared_float_vec normalized_input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(normalized_input_vec));

    // std::cout << "classification model prediction." << std::endl;
    // std::cout << "std vector: " << std::endl;
    // std::cout << input_std_.transpose() << std::endl;
    // std::cout << "mean vector: " << std::endl;
    // std::cout << input_mean_.transpose() << std::endl;
    // std::cout << "input vector: " << std::endl;
    // std::cout << input.transpose() << std::endl;
    // std::cout << "input dim: " << std::endl;
    // std::cout << input_dim_ << std::endl;

    auto result = model_->predict({fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, input_dim_), normalized_input_vec_ref)});

    if(isnan(result[0].get(0,0,0,0,0)))
    {
        std::cout << "std vector: " << std::endl;
        std::cout << input_std_.transpose() << std::endl;
        std::cout << "mean vector: " << std::endl;
        std::cout << input_mean_.transpose() << std::endl;
        std::cout << "input vector: " << std::endl;
        std::cout << input.transpose() << std::endl;
        std::cout << "normalized input: " << normalized_input.transpose() << std::endl;
        std::cout << "The prediction result is nan. Error!" << std::endl;
        getchar();
    }

    // std::cout << result[0].get(0,0,0,0,0) << std::endl;

    return result[0].get(0,0,0,0,0);
}

Eigen::VectorXd RegressionModel::predict(Eigen::VectorXd input)
{
    Eigen::VectorXd normalized_input = (input - input_mean_).cwiseQuotient(input_std_);

    fdeep::float_vec normalized_input_vec(normalized_input.data(), normalized_input.data() + normalized_input.size());
    fdeep::shared_float_vec normalized_input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(normalized_input_vec));

    // std::cout << "regression model prediction." << std::endl;
    // std::cout << "std vector: " << std::endl;
    // std::cout << input_std_.transpose() << std::endl;
    // std::cout << "mean vector: " << std::endl;
    // std::cout << input_mean_.transpose() << std::endl;
    // std::cout << "input vector: " << std::endl;
    // std::cout << input.transpose() << std::endl;
    // std::cout << "input dim: " << std::endl;
    // std::cout << input_dim_ << std::endl;

    // auto time_before_dynamics_prediction = std::chrono::high_resolution_clock::now();
    auto result = model_->predict({fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, input_dim_), normalized_input_vec_ref)});
    // auto time_after_dynamics_prediction = std::chrono::high_resolution_clock::now();
    // std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_dynamics_prediction - time_before_dynamics_prediction).count()/1000.0 << " ms" << std::endl;


    // fdeep::float_vec output_vec = *result[0].as_vector();

    Eigen::VectorXd normalized_output(output_dim_);

    for(int i = 0; i < output_dim_; i++)
    {
        normalized_output[i] = result[0].get(0,0,0,0,i);
    }

    Eigen::VectorXd output = normalized_output.cwiseProduct(output_std_) + output_mean_;

    return output;
}

inline void null_logger(const std::string& str)
{
}

NeuralNetworkInterface::NeuralNetworkInterface(std::string contact_transition_regression_model_file_path,
                                               std::string contact_transition_classification_model_file_path,
                                               std::string zero_step_capturability_classification_model_file_path,
                                               std::string one_step_capturability_classification_model_file_path)
{
    // set up the contact transition feasibility classifier and objective regressor
    for(int contact_status_code_int = 0; contact_status_code_int < 10; contact_status_code_int++)
    {
        ContactTransitionCode contact_status_code = static_cast<ContactTransitionCode>(contact_status_code_int);

        // load the regression neural network
        std::string regression_model_parameter_string = "_0.0005_256_0.0";
        std::shared_ptr<fdeep::model> dynamics_cost_regression_model = std::make_shared<fdeep::model>(fdeep::load_model(contact_transition_regression_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".json", false, null_logger));
        // std::shared_ptr<fdeep::model> dynamics_cost_regression_model = std::make_shared<fdeep::model>(fdeep::load_model(contact_transition_regression_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".json"));

        auto objective_regression_input_mean_std = readMeanStd(contact_transition_regression_model_file_path + "input_mean_std_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".txt");
        auto objective_regression_output_mean_std = readMeanStd(contact_transition_regression_model_file_path + "output_mean_std_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".txt");

        contact_transition_dynamics_cost_regression_models_map_.insert(std::make_pair(contact_status_code, RegressionModel(objective_regression_input_mean_std.first,
                                                                                                                           objective_regression_input_mean_std.second,
                                                                                                                           objective_regression_output_mean_std.first,
                                                                                                                           objective_regression_output_mean_std.second,
                                                                                                                           dynamics_cost_regression_model)));


        // // load the classification neural network
        // std::string calssification_model_parameter_string = "_0.0001_256_0.1";
        // std::shared_ptr<fdeep::model> feasibility_calssification_model = std::make_shared<fdeep::model>(fdeep::load_model(contact_transition_classification_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + calssification_model_parameter_string + ".json", false, null_logger));

        // auto feasibility_classification_input_mean_std = readMeanStd(contact_transition_classification_model_file_path + "input_mean_std_" + std::to_string(contact_status_code_int) + calssification_model_parameter_string + ".txt");

        // contact_transition_feasibility_calssification_models_map_.insert(std::make_pair(contact_status_code, ClassificationModel(feasibility_classification_input_mean_std.first,
        //                                                                                                                          feasibility_classification_input_mean_std.second,
        //                                                                                                                          feasibility_calssification_model)));
    }

    // set up the zero step capturability classifier
    for(int zero_step_capture_code_int = 0; zero_step_capture_code_int < 1; zero_step_capture_code_int++)
    {
        ZeroStepCaptureCode zero_step_capture_code = static_cast<ZeroStepCaptureCode>(zero_step_capture_code_int);

        // load the classification neural network
        // std::string calssification_model_parameter_string = "_0.0001_224_0.0"; // small disturbance
        std::string calssification_model_parameter_string = "_5e-05_256_0.1";
        std::shared_ptr<fdeep::model> zero_step_capturability_calssification_model = std::make_shared<fdeep::model>(fdeep::load_model(zero_step_capturability_classification_model_file_path + "zero_step_capture_nn_model_" + std::to_string(zero_step_capture_code_int) + calssification_model_parameter_string + ".json", false, null_logger));
        // std::shared_ptr<fdeep::model> zero_step_capturability_calssification_model = std::make_shared<fdeep::model>(fdeep::load_model(zero_step_capturability_classification_model_file_path + "zero_step_capture_nn_model_" + std::to_string(zero_step_capture_code_int) + calssification_model_parameter_string + ".json"));

        auto zero_step_capture_classification_input_mean_std = readMeanStd(zero_step_capturability_classification_model_file_path + "zero_step_capture_input_mean_std_" + std::to_string(zero_step_capture_code_int) + calssification_model_parameter_string + ".txt");

        zero_step_capturability_calssification_models_map_.insert(std::make_pair(zero_step_capture_code, ClassificationModel(zero_step_capture_classification_input_mean_std.first,
                                                                                                                             zero_step_capture_classification_input_mean_std.second,
                                                                                                                             zero_step_capturability_calssification_model)));
    }

    // set up the one step capturability classifier
    for(int one_step_capture_code_int = 0; one_step_capture_code_int < 3; one_step_capture_code_int++)
    {
        OneStepCaptureCode one_step_capture_code = static_cast<OneStepCaptureCode>(one_step_capture_code_int);

        // load the classification neural network
        // std::string calssification_model_parameter_string = "_0.0001_256_0.0"; // small disturbance
        std::string calssification_model_parameter_string = "_5e-05_256_0.1";
        std::shared_ptr<fdeep::model> one_step_capturability_calssification_model = std::make_shared<fdeep::model>(fdeep::load_model(one_step_capturability_classification_model_file_path + "one_step_capture_nn_model_" + std::to_string(one_step_capture_code_int) + calssification_model_parameter_string + ".json", false, null_logger));
        // std::shared_ptr<fdeep::model> one_step_capturability_calssification_model = std::make_shared<fdeep::model>(fdeep::load_model(one_step_capturability_classification_model_file_path + "one_step_capture_nn_model_" + std::to_string(one_step_capture_code_int) + calssification_model_parameter_string + ".json"));

        auto one_step_capture_classification_input_mean_std = readMeanStd(one_step_capturability_classification_model_file_path + "one_step_capture_input_mean_std_" + std::to_string(one_step_capture_code_int) + calssification_model_parameter_string + ".txt");

        one_step_capturability_calssification_models_map_.insert(std::make_pair(one_step_capture_code, ClassificationModel(one_step_capture_classification_input_mean_std.first,
                                                                                                                           one_step_capture_classification_input_mean_std.second,
                                                                                                                           one_step_capturability_calssification_model)));
    }
}

// bool NeuralNetworkInterface::predictFeasibility(std::shared_ptr<ContactState> branching_state)
// {
//     return true;
// }

std::pair<Eigen::VectorXd, Eigen::VectorXd> NeuralNetworkInterface::readMeanStd(std::string file_path)
{
    std::ifstream f_mean_std;
    f_mean_std.open(file_path, std::ifstream::in);

    Eigen::VectorXd mean_eigen, std_eigen;
    double mean, std;
    std::vector<double> mean_data, std_data;

    while(f_mean_std >> mean)
    {
        mean_data.push_back(mean);
        f_mean_std >> std;
        std_data.push_back(std);
    }
    f_mean_std.close();

    int dim = mean_data.size();
    mean_eigen.resize(dim);
    std_eigen.resize(dim);

    for(int i = 0; i < dim; i++)
    {
        mean_eigen[i] = mean_data[i];
        std_eigen[i] = std_data[i];
    }

    return std::make_pair(mean_eigen, std_eigen);
}

Eigen::VectorXd NeuralNetworkInterface::constructFeatureVector(std::vector<RPYTF>& contact_manip_pose_vec, Translation3D& com, Vector3D& com_dot)
{
    unsigned int pose_feature_size = contact_manip_pose_vec.size()*6;
    Eigen::VectorXd feature_vector(pose_feature_size+6);

    unsigned int counter = 0;
    for(auto & contact_pose : contact_manip_pose_vec)
    {
        feature_vector[counter]   = contact_pose.x_;
        feature_vector[counter+1] = contact_pose.y_;
        feature_vector[counter+2] = contact_pose.z_;
        feature_vector[counter+3] = contact_pose.roll_ * DEG2RAD;
        feature_vector[counter+4] = contact_pose.pitch_ * DEG2RAD;
        feature_vector[counter+5] = contact_pose.yaw_ * DEG2RAD;

        counter += 6;
    }

    feature_vector.block(pose_feature_size, 0, 3, 1) = com.cast<double>();
    feature_vector.block(pose_feature_size+3, 0, 3, 1) = com_dot.cast<double>();

    return feature_vector;
}

std::tuple<bool, float, Translation3D, Vector3D> NeuralNetworkInterface::predictContactTransitionDynamicsCost(std::shared_ptr<ContactState> branching_state)
{
    TransformationMatrix feet_mean_transform = branching_state->parent_->getFeetMeanTransform();

    std::shared_ptr<ContactState> standard_input_state = branching_state->getStandardInputState(DynOptApplication::CONTACT_TRANSITION_DYNOPT);
    std::shared_ptr<ContactState> prev_state = standard_input_state->parent_;

    // decide the contact transition code & the poses
    auto transition_code_poses_pair = standard_input_state->getTransitionCodeAndPoses();
    ContactTransitionCode contact_transition_code = transition_code_poses_pair.first;
    std::vector<RPYTF> contact_manip_pose_vec = transition_code_poses_pair.second;

    Eigen::VectorXd feature_vector = constructFeatureVector(contact_manip_pose_vec, prev_state->com_, prev_state->com_dot_);

    // std::cout << "======================" << std::endl;
    // std::cout << "feature vector: " << std::endl;

    // for(unsigned int i = 0; i < feature_vector.size(); i++)
    // {
    //     std::cout << feature_vector[i] << ", ";
    // }
    // std::cout << std::endl;

    // decide whether the transition is dynamically feasible
    bool dynamics_feasibility;

    // float dynamics_feasibility_prediction = contact_transition_feasibility_calssification_models_map_.find(contact_transition_code)->second.predict(feature_vector);
    // dynamics_feasibility = (dynamics_feasibility_prediction >= 0.5);

    // if(contact_transition_code == ContactTransitionCode::FEET_AND_ONE_HAND_BREAK_HAND || contact_transition_code == ContactTransitionCode::FEET_AND_TWO_HANDS_BREAK_HAND)
    // {
    //     dynamics_feasibility = true;
    // }

    dynamics_feasibility = true;

    if(dynamics_feasibility)
    {
        Eigen::VectorXd prediction = contact_transition_dynamics_cost_regression_models_map_.find(contact_transition_code)->second.predict(feature_vector);

        Translation3D predicted_com = prediction.block(0,0,3,1).cast<float>();
        Vector3D predicted_com_dot = prediction.block(3,0,3,1).cast<float>();

        // std::cout << "prediction: " << prediction.transpose() << std::endl;

        if(branching_state->prev_move_manip_ == ContactManipulator::R_LEG || branching_state->prev_move_manip_ == ContactManipulator::R_ARM)
        {
            // std::cout << feet_mean_transform << std::endl;
            // std::cout << prev_state->getFeetMeanTransform() << std::endl;
            predicted_com[1] = -predicted_com[1];
            predicted_com_dot[1] = -predicted_com_dot[1];
        }

        // getchar();

        Translation3D final_com = (feet_mean_transform * predicted_com.homogeneous()).block(0,0,3,1);
        Vector3D final_com_dot = feet_mean_transform.block(0,0,3,3) * predicted_com_dot;
        float dynamics_cost = prediction[6];

        return make_tuple(dynamics_feasibility, dynamics_cost, final_com, final_com_dot);
    }
    else
    {
        Translation3D dummy_com(0,0,0);
        Vector3D dummy_com_dot(0,0,0);
        float dummy_dynamics_cost = 0;

        return make_tuple(dynamics_feasibility, dummy_dynamics_cost, dummy_com, dummy_com_dot);
    }
}

bool NeuralNetworkInterface::predictContactTransitionDynamics(std::shared_ptr<ContactState> branching_state, float& dynamics_cost)
{
    std::tuple<bool, float, Translation3D, Vector3D> dynamics_prediction = predictContactTransitionDynamicsCost(branching_state);
    bool dynamics_feasibility = std::get<0>(dynamics_prediction);
    dynamics_cost = std::get<1>(dynamics_prediction);
    dynamics_cost = std::max(dynamics_cost,float(0.0));
    branching_state->com_ = std::get<2>(dynamics_prediction);
    branching_state->com_dot_ = std::get<3>(dynamics_prediction);

    return dynamics_feasibility;
}

bool NeuralNetworkInterface::predictZeroStepCaptureDynamics(std::shared_ptr<ContactState> zero_step_capture_state)
{
    std::shared_ptr<ContactState> standard_input_state = zero_step_capture_state->getStandardInputState(DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT);

    // decide the motion code & the poses
    auto motion_code_poses_pair = standard_input_state->getZeroStepCapturabilityCodeAndPoses();
    ZeroStepCaptureCode zero_step_capture_code = motion_code_poses_pair.first;
    std::vector<RPYTF> contact_manip_pose_vec = motion_code_poses_pair.second;

    Eigen::VectorXd feature_vector = constructFeatureVector(contact_manip_pose_vec, standard_input_state->com_, standard_input_state->lmom_);

    // std::cout << "Zero Step Capture Code: " << zero_step_capture_code << std::endl;

    // decide whether it is zero step capturable
    float zero_step_capturability_prediction = zero_step_capturability_calssification_models_map_.find(zero_step_capture_code)->second.predict(feature_vector);
    bool zero_step_capturable = (zero_step_capturability_prediction >= 0.5);

    return zero_step_capturable;
}

bool NeuralNetworkInterface::predictOneStepCaptureDynamics(std::shared_ptr<ContactState> one_step_capture_state)
{
    std::shared_ptr<ContactState> standard_input_state = one_step_capture_state->getStandardInputState(DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);
    std::shared_ptr<ContactState> prev_state = standard_input_state->parent_;

    // decide the motion code & the poses
    auto motion_code_poses_pair = standard_input_state->getOneStepCapturabilityCodeAndPoses();
    OneStepCaptureCode one_step_capture_code = motion_code_poses_pair.first;
    std::vector<RPYTF> contact_manip_pose_vec = motion_code_poses_pair.second;

    Eigen::VectorXd feature_vector = constructFeatureVector(contact_manip_pose_vec, prev_state->com_, prev_state->lmom_);

    // std::cout << "One Step Capture Code: " << one_step_capture_code << std::endl;
    // std::cout << feature_vector.transpose() << std::endl;

    // decide whether it is one step capturable
    float one_step_capturability_prediction = one_step_capturability_calssification_models_map_.find(one_step_capture_code)->second.predict(feature_vector);
    bool one_step_capturable = (one_step_capturability_prediction >= 0.5);

    // if(one_step_capture_code == 8)
    // {
    //     std::cout << feature_vector.transpose() << std::endl;
    //     std::cout << one_step_capturability_prediction << std::endl;
    //     // std::cout << "Capturable: " << one_step_capturable << std::endl;
    //     getchar();
    //     // one_step_capturable = true;
    // }

    return one_step_capturable;
}

Eigen::VectorXd NeuralNetworkInterface::getOneStepCaptureFeatureVector(std::shared_ptr<ContactState> one_step_capture_state)
{
    std::shared_ptr<ContactState> standard_input_state = one_step_capture_state->getStandardInputState(DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);
    std::shared_ptr<ContactState> prev_state = standard_input_state->parent_;

    // decide the motion code & the poses
    auto motion_code_poses_pair = standard_input_state->getOneStepCapturabilityCodeAndPoses();
    OneStepCaptureCode one_step_capture_code = motion_code_poses_pair.first;
    std::vector<RPYTF> contact_manip_pose_vec = motion_code_poses_pair.second;

    Eigen::VectorXd feature_vector = constructFeatureVector(contact_manip_pose_vec, prev_state->com_, prev_state->lmom_);

    return feature_vector;
}