
#include "Utilities.hpp"

float ClassificationModel::predict(Eigen::VectorXd input, NeuralNetworkModelType model_type)
{
    Eigen::VectorXd normalized_input = (input - input_mean_).cwiseQuotient(input_std_);
    float result;

    if(model_type == NeuralNetworkModelType::FRUGALLY_DEEP)
    {
        fdeep::float_vec normalized_input_vec(normalized_input.data(), normalized_input.data() + normalized_input.size());
        fdeep::shared_float_vec normalized_input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(normalized_input_vec));
        auto fdeep_result = fdeep_model_->predict({fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, input_dim_), normalized_input_vec_ref)});
        result = fdeep_result[0].get(0,0,0,0,0);
    }
    else if(model_type == NeuralNetworkModelType::TENSORFLOW)
    {
        // Actually run the image through the model.
        std::vector<tensorflow::Tensor> tf_result;
        tensorflow::Tensor normalized_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, normalized_input.size()}));
        auto normalized_input_eigen_tensor = normalized_input_tensor.matrix<float>();
        for(int i = 0; i < normalized_input.size(); i++)
        {
            normalized_input_eigen_tensor(0,i) = normalized_input[i];
        }
        tensorflow::Status run_status = tf_model_->Run({{"input_1", normalized_input_tensor}}, {"dense_3/Sigmoid"}, {}, &tf_result);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            getchar();
        }

        result = tf_result[0].matrix<float>()(0,0);
    }

    // std::cout << "classification model prediction." << std::endl;
    // std::cout << "std vector: " << std::endl;
    // std::cout << input_std_.transpose() << std::endl;
    // std::cout << "mean vector: " << std::endl;
    // std::cout << input_mean_.transpose() << std::endl;
    // std::cout << "input vector: " << std::endl;
    // std::cout << input.transpose() << std::endl;
    // std::cout << "input dim: " << std::endl;
    // std::cout << input_dim_ << std::endl;

    if(isnan(result))
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

    // std::cout << result << std::endl;

    return result;
}

std::vector<float> ClassificationModel::predict(Eigen::MatrixXd input, NeuralNetworkModelType model_type)
{
    size_t data_num = input.cols();
    Eigen::MatrixXd normalized_input = (input - input_mean_.replicate(1,data_num)).cwiseQuotient(input_std_.replicate(1,data_num));
    std::vector<float> result(data_num);

    if(model_type == NeuralNetworkModelType::FRUGALLY_DEEP)
    {
        std::vector<fdeep::tensor5s> tensor_vector;
        tensor_vector.reserve(data_num);

        for(int data_id = 0; data_id < data_num; data_id++)
        {
            fdeep::float_vec normalized_input_vec(normalized_input.col(data_id).data(), normalized_input.col(data_id).data() + normalized_input.col(data_id).size());
            fdeep::shared_float_vec normalized_input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(normalized_input_vec));
            tensor_vector.push_back({fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, input_dim_), normalized_input_vec_ref)});
        }

        // auto fdeep_result = fdeep_model_->predict(tensor_vector);
        auto fdeep_result = fdeep_model_->predict_multi(tensor_vector, false);

        for(int data_id = 0; data_id < data_num; data_id++)
        {
            result[data_id] = fdeep_result[data_id][0].get(0,0,0,0,0);
        }
    }
    else if(model_type == NeuralNetworkModelType::TENSORFLOW)
    {
        // Actually run the image through the model.
        std::vector<tensorflow::Tensor> tf_result;
        tensorflow::Tensor normalized_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({normalized_input.cols(), normalized_input.rows()}));
        auto normalized_input_eigen_tensor = normalized_input_tensor.matrix<float>();
        for(int data_id = 0; data_id < data_num; data_id++)
        {
            for(int i = 0; i < normalized_input.rows(); i++)
            {
                normalized_input_eigen_tensor(data_id,i) = normalized_input(i,data_id);
            }
        }

        tensorflow::Status run_status = tf_model_->Run({{"input_1", normalized_input_tensor}}, {"dense_3/Sigmoid"}, {}, &tf_result);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            getchar();
        }

        for(int data_id = 0; data_id < data_num; data_id++)
        {
            result[data_id] = tf_result[0].matrix<float>()(0,data_id);
        }
    }

    // std::cout << "classification model prediction." << std::endl;
    // std::cout << "std vector: " << std::endl;
    // std::cout << input_std_.transpose() << std::endl;
    // std::cout << "mean vector: " << std::endl;
    // std::cout << input_mean_.transpose() << std::endl;
    // std::cout << "input vector: " << std::endl;
    // std::cout << input.transpose() << std::endl;
    // std::cout << "input dim: " << std::endl;
    // std::cout << input_dim_ << std::endl;


    for(int data_id = 0; data_id < data_num; data_id++)
    {
        if(isnan(result[data_id]))
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
    }

    return result;
}

Eigen::VectorXd RegressionModel::predict(Eigen::VectorXd input, NeuralNetworkModelType model_type)
{
    Eigen::VectorXd normalized_input = (input - input_mean_).cwiseQuotient(input_std_);
    Eigen::VectorXd normalized_result(output_dim_);

    if(model_type == NeuralNetworkModelType::FRUGALLY_DEEP)
    {
        fdeep::float_vec normalized_input_vec(normalized_input.data(), normalized_input.data() + normalized_input.size());
        fdeep::shared_float_vec normalized_input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(normalized_input_vec));

        // auto time_before_dynamics_prediction = std::chrono::high_resolution_clock::now();
        auto fdeep_result = fdeep_model_->predict({fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, input_dim_), normalized_input_vec_ref)});
        // auto time_after_dynamics_prediction = std::chrono::high_resolution_clock::now();
        // std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_dynamics_prediction - time_before_dynamics_prediction).count()/1000.0 << " ms" << std::endl;

        // fdeep::float_vec output_vec = *fdeep_result[0].as_vector();

        for(int i = 0; i < output_dim_; i++)
        {
            normalized_result[i] = fdeep_result[0].get(0,0,0,0,i);
        }
    }
    else if(model_type == NeuralNetworkModelType::TENSORFLOW)
    {

    }

    Eigen::VectorXd result = normalized_result.cwiseProduct(output_std_) + output_mean_;

    // std::cout << "regression model prediction." << std::endl;
    // std::cout << "input vector: " << std::endl;
    // std::cout << input.transpose() << std::endl;
    // std::cout << "input std vector: " << std::endl;
    // std::cout << input_std_.transpose() << std::endl;
    // std::cout << "input mean vector: " << std::endl;
    // std::cout << input_mean_.transpose() << std::endl;
    // std::cout << "input dim: " << std::endl;
    // std::cout << input_dim_ << std::endl;
    // std::cout << "output std vector: " << std::endl;
    // std::cout << output_std_.transpose() << std::endl;
    // std::cout << "output mean vector: " << std::endl;
    // std::cout << output_mean_.transpose() << std::endl;
    // std::cout << "output dim: " << std::endl;
    // std::cout << output_dim_ << std::endl;
    // std::cout << "result: " << std::endl;
    // std::cout << result.transpose() << std::endl;

    return result;
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
    for(int contact_status_code_int = 0; contact_status_code_int < 1; contact_status_code_int++)
    {
        ContactTransitionCode contact_status_code = static_cast<ContactTransitionCode>(contact_status_code_int);

        // load the regression neural network
        std::string regression_model_parameter_string = "_0.0001_256_0.0";

        // load frugally-deep model
        std::shared_ptr<fdeep::model> dynamics_cost_regression_fdeep_model = std::make_shared<fdeep::model>(fdeep::load_model(contact_transition_regression_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".json", false, null_logger));
        // std::shared_ptr<fdeep::model> dynamics_cost_regression_fdeep_model = std::make_shared<fdeep::model>(fdeep::load_model(contact_transition_regression_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".json"));

        // // load tensorflow model
        // std::shared_ptr<tensorflow::Session> dynamics_cost_regression_tf_model;
        // tensorflow::Status load_graph_status = LoadTensorflowGraph(contact_transition_regression_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".pb", &dynamics_cost_regression_tf_model);
        // if (!load_graph_status.ok())
        // {
        //     LOG(ERROR) << load_graph_status;
        //     getchar();
        // }

        auto objective_regression_input_mean_std = readMeanStd(contact_transition_regression_model_file_path + "input_mean_std_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".txt");
        auto objective_regression_output_mean_std = readMeanStd(contact_transition_regression_model_file_path + "output_mean_std_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".txt");

        contact_transition_dynamics_cost_regression_models_map_.insert(std::make_pair(contact_status_code, RegressionModel(objective_regression_input_mean_std.first,
                                                                                                                           objective_regression_input_mean_std.second,
                                                                                                                           objective_regression_output_mean_std.first,
                                                                                                                           objective_regression_output_mean_std.second,
                                                                                                                           dynamics_cost_regression_fdeep_model,
                                                                                                                           nullptr)));


        // // load the classification neural network
        // std::string calssification_model_parameter_string = "_0.0001_256_0.1";
        // std::shared_ptr<fdeep::model> feasibility_calssification_model = std::make_shared<fdeep::model>(fdeep::load_model(contact_transition_classification_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + calssification_model_parameter_string + ".json", false, null_logger));

        // auto feasibility_classification_input_mean_std = readMeanStd(contact_transition_classification_model_file_path + "input_mean_std_" + std::to_string(contact_status_code_int) + calssification_model_parameter_string + ".txt");

        // contact_transition_feasibility_calssification_models_map_.insert(std::make_pair(contact_status_code, ClassificationModel(feasibility_classification_input_mean_std.first,
        //                                                                                                                          feasibility_classification_input_mean_std.second,
        //                                                                                                                          feasibility_calssification_model,
        //                                                                                                                          nullptr)));
    }

    // set up the zero step capturability classifier
    for(int zero_step_capture_code_int = 0; zero_step_capture_code_int < 1; zero_step_capture_code_int++)
    {
        ZeroStepCaptureCode zero_step_capture_code = static_cast<ZeroStepCaptureCode>(zero_step_capture_code_int);

        // load the classification neural network
        // std::string calssification_model_parameter_string = "_0.0001_224_0.0"; // small disturbance
        std::string calssification_model_parameter_string = "_5e-05_256_0.1";

        // load frugally-deep model
        std::shared_ptr<fdeep::model> zero_step_capturability_calssification_fdeep_model = std::make_shared<fdeep::model>(fdeep::load_model(zero_step_capturability_classification_model_file_path + "zero_step_capture_nn_model_" + std::to_string(zero_step_capture_code_int) + calssification_model_parameter_string + ".json", false, null_logger));
        // std::shared_ptr<fdeep::model> zero_step_capturability_calssification_fdeep_model = std::make_shared<fdeep::model>(fdeep::load_model(zero_step_capturability_classification_model_file_path + "zero_step_capture_nn_model_" + std::to_string(zero_step_capture_code_int) + calssification_model_parameter_string + ".json"));

        // load tensorflow model
        std::shared_ptr<tensorflow::Session> zero_step_capturability_calssification_tf_model;
        tensorflow::Status load_graph_status = LoadTensorflowGraph(zero_step_capturability_classification_model_file_path + "zero_step_capture_nn_model_" + std::to_string(zero_step_capture_code_int) + calssification_model_parameter_string + ".pb", &zero_step_capturability_calssification_tf_model);
        if (!load_graph_status.ok())
        {
            LOG(ERROR) << load_graph_status;
            getchar();
        }

        auto zero_step_capture_classification_input_mean_std = readMeanStd(zero_step_capturability_classification_model_file_path + "zero_step_capture_input_mean_std_" + std::to_string(zero_step_capture_code_int) + calssification_model_parameter_string + ".txt");

        zero_step_capturability_calssification_models_map_.insert(std::make_pair(zero_step_capture_code, ClassificationModel(zero_step_capture_classification_input_mean_std.first,
                                                                                                                             zero_step_capture_classification_input_mean_std.second,
                                                                                                                             zero_step_capturability_calssification_fdeep_model,
                                                                                                                             zero_step_capturability_calssification_tf_model)));
    }

    // set up the one step capturability classifier
    for(int one_step_capture_code_int = 0; one_step_capture_code_int < 3; one_step_capture_code_int++)
    {
        OneStepCaptureCode one_step_capture_code = static_cast<OneStepCaptureCode>(one_step_capture_code_int);

        // load the classification neural network
        // std::string calssification_model_parameter_string = "_0.0001_256_0.0"; // small disturbance
        std::string calssification_model_parameter_string = "_5e-05_256_0.1";

        // load frugally-deep model
        std::shared_ptr<fdeep::model> one_step_capturability_calssification_fdeep_model = std::make_shared<fdeep::model>(fdeep::load_model(one_step_capturability_classification_model_file_path + "one_step_capture_nn_model_" + std::to_string(one_step_capture_code_int) + calssification_model_parameter_string + ".json", false, null_logger));
        // std::shared_ptr<fdeep::model> one_step_capturability_calssification_fdeep_model = std::make_shared<fdeep::model>(fdeep::load_model(one_step_capturability_classification_model_file_path + "one_step_capture_nn_model_" + std::to_string(one_step_capture_code_int) + calssification_model_parameter_string + ".json"));

        // load tensorflow model
        std::shared_ptr<tensorflow::Session> one_step_capturability_calssification_tf_model;
        tensorflow::Status load_graph_status = LoadTensorflowGraph(one_step_capturability_classification_model_file_path + "one_step_capture_nn_model_" + std::to_string(one_step_capture_code_int) + calssification_model_parameter_string + ".pb", &one_step_capturability_calssification_tf_model);
        if (!load_graph_status.ok())
        {
            LOG(ERROR) << load_graph_status;
            getchar();
        }

        auto one_step_capture_classification_input_mean_std = readMeanStd(one_step_capturability_classification_model_file_path + "one_step_capture_input_mean_std_" + std::to_string(one_step_capture_code_int) + calssification_model_parameter_string + ".txt");

        one_step_capturability_calssification_models_map_.insert(std::make_pair(one_step_capture_code, ClassificationModel(one_step_capture_classification_input_mean_std.first,
                                                                                                                           one_step_capture_classification_input_mean_std.second,
                                                                                                                           one_step_capturability_calssification_fdeep_model,
                                                                                                                           one_step_capturability_calssification_tf_model)));
    }
}

// bool NeuralNetworkInterface::predictFeasibility(std::shared_ptr<ContactState> branching_state)
// {
//     return true;
// }

tensorflow::Status NeuralNetworkInterface::LoadTensorflowGraph(const std::string graph_file_name, std::shared_ptr<tensorflow::Session>* session)
{
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);

    if (!load_graph_status.ok())
    {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }

    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status session_create_status = (*session)->Create(graph_def);

    if (!session_create_status.ok())
    {
        return session_create_status;
    }

    return tensorflow::Status::OK();
}

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

std::tuple<bool, float, Translation3D, Vector3D> NeuralNetworkInterface::predictContactTransitionDynamicsCost(std::shared_ptr<ContactState> branching_state, NeuralNetworkModelType model_type)
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

    // float dynamics_feasibility_prediction = contact_transition_feasibility_calssification_models_map_.find(contact_transition_code)->second.predict(feature_vector, model_type);
    // dynamics_feasibility = (dynamics_feasibility_prediction >= 0.5);

    // if(contact_transition_code == ContactTransitionCode::FEET_AND_ONE_HAND_BREAK_HAND || contact_transition_code == ContactTransitionCode::FEET_AND_TWO_HANDS_BREAK_HAND)
    // {
    //     dynamics_feasibility = true;
    // }

    dynamics_feasibility = true;

    if(dynamics_feasibility)
    {
        Eigen::VectorXd prediction = contact_transition_dynamics_cost_regression_models_map_.find(contact_transition_code)->second.predict(feature_vector, model_type);

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

bool NeuralNetworkInterface::predictContactTransitionDynamics(std::shared_ptr<ContactState> branching_state, float& dynamics_cost, NeuralNetworkModelType model_type)
{
    std::tuple<bool, float, Translation3D, Vector3D> dynamics_prediction = predictContactTransitionDynamicsCost(branching_state, model_type);
    bool dynamics_feasibility = std::get<0>(dynamics_prediction);
    dynamics_cost = std::get<1>(dynamics_prediction);
    dynamics_cost = std::max(dynamics_cost,float(0.0));
    branching_state->com_ = std::get<2>(dynamics_prediction);
    branching_state->com_dot_ = std::get<3>(dynamics_prediction);

    return dynamics_feasibility;
}

bool NeuralNetworkInterface::predictZeroStepCaptureDynamics(std::shared_ptr<ContactState> zero_step_capture_state, NeuralNetworkModelType model_type)
{
    std::shared_ptr<ContactState> standard_input_state = zero_step_capture_state->getStandardInputState(DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT);

    // decide the motion code & the poses
    auto motion_code_poses_pair = standard_input_state->getZeroStepCapturabilityCodeAndPoses();
    ZeroStepCaptureCode zero_step_capture_code = motion_code_poses_pair.first;
    std::vector<RPYTF> contact_manip_pose_vec = motion_code_poses_pair.second;

    Eigen::VectorXd feature_vector = constructFeatureVector(contact_manip_pose_vec, standard_input_state->com_, standard_input_state->lmom_);

    // std::cout << "Zero Step Capture Code: " << zero_step_capture_code << std::endl;

    // decide whether it is zero step capturable
    float zero_step_capturability_prediction = zero_step_capturability_calssification_models_map_.find(zero_step_capture_code)->second.predict(feature_vector, model_type);
    bool zero_step_capturable = (zero_step_capturability_prediction >= 0.5);

    return zero_step_capturable;
}

std::vector<bool> NeuralNetworkInterface::predictZeroStepCaptureDynamics(std::vector< std::shared_ptr<ContactState> > zero_step_capture_state_vec, NeuralNetworkModelType model_type)
{
    std::vector<bool> zero_step_capturability_vec(zero_step_capture_state_vec.size());
    std::unordered_map<ZeroStepCaptureCode, std::vector<int>, EnumClassHash> capture_code_zero_step_capture_state_indices_map;
    std::vector<Eigen::VectorXd> feature_vector_vec(zero_step_capture_state_vec.size());

    if(zero_step_capture_state_vec.size() != 0)
    {
        int data_id = 0;
        for(auto zero_step_capture_state : zero_step_capture_state_vec)
        {
            // get reference frame
            std::shared_ptr<ContactState> standard_input_state = zero_step_capture_state->getStandardInputState(DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT);

            // decide the motion code & the poses
            auto motion_code_poses_pair = standard_input_state->getZeroStepCapturabilityCodeAndPoses();
            ZeroStepCaptureCode zero_step_capture_code = motion_code_poses_pair.first;
            std::vector<RPYTF> contact_manip_pose_vec = motion_code_poses_pair.second;

            if(capture_code_zero_step_capture_state_indices_map.find(zero_step_capture_code) == capture_code_zero_step_capture_state_indices_map.end())
            {
                capture_code_zero_step_capture_state_indices_map[zero_step_capture_code] = {data_id};
            }
            else
            {
                capture_code_zero_step_capture_state_indices_map[zero_step_capture_code].push_back(data_id);
            }

            feature_vector_vec[data_id] = constructFeatureVector(contact_manip_pose_vec, standard_input_state->com_, standard_input_state->lmom_);

            data_id++;
        }

        for(auto & capture_code_zero_step_capture_state_index_pair : capture_code_zero_step_capture_state_indices_map)
        {
            ZeroStepCaptureCode zero_step_capture_code = capture_code_zero_step_capture_state_index_pair.first;
            int input_dim = feature_vector_vec[capture_code_zero_step_capture_state_index_pair.second[0]].size();
            int data_num = capture_code_zero_step_capture_state_index_pair.second.size();

            Eigen::MatrixXd feature_matrix(input_dim, data_num);

            int query_data_id = 0;
            for(auto & data_id : capture_code_zero_step_capture_state_index_pair.second)
            {
                feature_matrix.col(query_data_id) = feature_vector_vec[data_id];
                query_data_id++;
            }

            std::vector<float> zero_step_capturability_predictions = zero_step_capturability_calssification_models_map_.find(zero_step_capture_code)->second.predict(feature_matrix, model_type);

            query_data_id = 0;
            for(auto & data_id : capture_code_zero_step_capture_state_index_pair.second)
            {
                zero_step_capturability_vec[data_id] = zero_step_capturability_predictions[query_data_id] >= 0.5;
                query_data_id++;
            }
        }
    }

    return zero_step_capturability_vec;
}

bool NeuralNetworkInterface::predictOneStepCaptureDynamics(std::shared_ptr<ContactState> one_step_capture_state, NeuralNetworkModelType model_type)
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
    float one_step_capturability_prediction = one_step_capturability_calssification_models_map_.find(one_step_capture_code)->second.predict(feature_vector, model_type);
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

std::vector<bool> NeuralNetworkInterface::predictOneStepCaptureDynamics(std::vector< std::shared_ptr<ContactState> > one_step_capture_state_vec, NeuralNetworkModelType model_type)
{
    std::vector<bool> one_step_capturability_vec(one_step_capture_state_vec.size());
    std::unordered_map<OneStepCaptureCode, std::vector<int>, EnumClassHash> capture_code_one_step_capture_state_indices_map;
    std::vector<Eigen::VectorXd> feature_vector_vec(one_step_capture_state_vec.size());

    if(one_step_capture_state_vec.size() != 0)
    {
        int data_id = 0;
        for(auto one_step_capture_state : one_step_capture_state_vec)
        {
            // get reference frame
            std::shared_ptr<ContactState> standard_input_state = one_step_capture_state->getStandardInputState(DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);
            std::shared_ptr<ContactState> prev_state = standard_input_state->parent_;

            // decide the motion code & the poses
            auto motion_code_poses_pair = standard_input_state->getOneStepCapturabilityCodeAndPoses();
            OneStepCaptureCode one_step_capture_code = motion_code_poses_pair.first;
            std::vector<RPYTF> contact_manip_pose_vec = motion_code_poses_pair.second;

            if(capture_code_one_step_capture_state_indices_map.find(one_step_capture_code) == capture_code_one_step_capture_state_indices_map.end())
            {
                capture_code_one_step_capture_state_indices_map[one_step_capture_code] = {data_id};
            }
            else
            {
                capture_code_one_step_capture_state_indices_map[one_step_capture_code].push_back(data_id);
            }

            feature_vector_vec[data_id] = constructFeatureVector(contact_manip_pose_vec, prev_state->com_, prev_state->lmom_);

            data_id++;
        }

        for(auto & capture_code_one_step_capture_state_index_pair : capture_code_one_step_capture_state_indices_map)
        {
            OneStepCaptureCode one_step_capture_code = capture_code_one_step_capture_state_index_pair.first;
            int input_dim = feature_vector_vec[capture_code_one_step_capture_state_index_pair.second[0]].size();
            int data_num = capture_code_one_step_capture_state_index_pair.second.size();

            Eigen::MatrixXd feature_matrix(input_dim, data_num);

            int query_data_id = 0;
            for(auto & data_id : capture_code_one_step_capture_state_index_pair.second)
            {
                feature_matrix.col(query_data_id) = feature_vector_vec[data_id];
                query_data_id++;
            }

            std::vector<float> one_step_capturability_predictions = one_step_capturability_calssification_models_map_.find(one_step_capture_code)->second.predict(feature_matrix, model_type);

            query_data_id = 0;
            for(auto & data_id : capture_code_one_step_capture_state_index_pair.second)
            {
                one_step_capturability_vec[data_id] = one_step_capturability_predictions[query_data_id] >= 0.5;
                query_data_id++;
            }
        }

    }

    return one_step_capturability_vec;
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