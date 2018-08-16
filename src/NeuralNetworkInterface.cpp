
#include "Utilities.hpp"

int ClassificationModel::predict(Eigen::VectorXd input)
{
    Eigen::VectorXd normalized_input = (input - input_mean_).cwiseQuotient(input_std_);

    fdeep::float_vec input_vec(input.data(), input.data() + input.size());
    fdeep::shared_float_vec input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(input_vec));

    fdeep::tensor3s result = model_->predict({fdeep::tensor3(fdeep::shape3(input_dim_, 1, 1), input_vec_ref)});

    return result[0].get(0,0,0);
}

Eigen::VectorXd RegressionModel::predict(Eigen::VectorXd input)
{
    Eigen::VectorXd normalized_input = (input - input_mean_).cwiseQuotient(input_std_);

    fdeep::float_vec normalized_input_vec(normalized_input.data(), normalized_input.data() + normalized_input.size());
    fdeep::shared_float_vec normalized_input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(normalized_input_vec));

    // std::cout << "std vector: " << std::endl;
    // std::cout << input_std_.transpose() << std::endl;
    // std::cout << "mean vector: " << std::endl;
    // std::cout << input_mean_.transpose() << std::endl;
    // std::cout << "input vector: " << std::endl;
    // std::cout << input.transpose() << std::endl;
    // std::cout << "input dim: " << std::endl;
    // std::cout << input_dim_ << std::endl;

    auto result = model_->predict({fdeep::tensor3(fdeep::shape3(input_dim_, 1, 1), normalized_input_vec_ref)});

    // fdeep::float_vec output_vec = *result[0].as_vector();

    Eigen::VectorXd normalized_output(output_dim_);

    for(int i = 0; i < output_dim_; i++)
    {
        normalized_output[i] = result[0].get(0,0,i);
    }

    Eigen::VectorXd output = normalized_output.cwiseProduct(output_std_) + output_mean_;

    return output;
}

NeuralNetworkInterface::NeuralNetworkInterface(std::string model_file_path)
{
    std::vector<ContactTransitionCode> state_transition_contact_status_code_vec = {ContactTransitionCode::FEET_ONLY_MOVE_FOOT};

    for(auto & contact_status_code : state_transition_contact_status_code_vec)
    {
        // load the neural network
        // std::shared_ptr<fdeep::model> feasibility_calssification_model = std::make_shared<fdeep::model>(fdeep::load_model(model_file_path + "feasibility_calssification_64_64_" + contact_status_code + ".json"));
        std::shared_ptr<fdeep::model> dynamics_cost_regression_model = std::make_shared<fdeep::model>(fdeep::load_model(model_file_path + "nn_model_" + std::to_string(int(contact_status_code)) + "_0.0005_192_0.0.json"));

        // load the input/output mean and std
        std::ifstream f_input_mean_std, f_output_mean_std;

        f_input_mean_std.open(model_file_path + "input_mean_std_" + std::to_string(int(contact_status_code)) + "_0.0005_192_0.0.txt",std::ifstream::in);
        f_output_mean_std.open(model_file_path + "output_mean_std_" + std::to_string(int(contact_status_code)) + "_0.0005_192_0.0.txt",std::ifstream::in);

        std::vector<double> input_mean, input_std, output_mean, output_std;
        double mean_data, std_data;

        while(f_input_mean_std >> mean_data)
        {
            input_mean.push_back(mean_data);
            f_input_mean_std >> std_data;
            input_std.push_back(std_data);
        }
        while(f_output_mean_std >> mean_data)
        {
            output_mean.push_back(mean_data);
            f_output_mean_std >> std_data;
            output_std.push_back(std_data);
        }

        f_input_mean_std.close();
        f_output_mean_std.close();

        int input_dim = input_mean.size();
        int output_dim = output_mean.size();

        Eigen::VectorXd input_mean_eigen(input_dim);
        Eigen::VectorXd input_std_eigen(input_dim);
        Eigen::VectorXd output_mean_eigen(output_dim);
        Eigen::VectorXd output_std_eigen(output_dim);

        for(int i = 0; i < input_dim; i++)
        {
            input_mean_eigen[i] = input_mean[i];
            input_std_eigen[i] = input_std[i];
        }
        for(int i = 0; i < output_dim; i++)
        {
            output_mean_eigen[i] = output_mean[i];
            output_std_eigen[i] = output_std[i];
        }

        // feasibility_calssification_models_map_.insert(std::make_pair(contact_status_code, ClassificationModel(input_mean_eigen, input_std_eigen, feasibility_calssification_model)));
        dynamics_cost_regression_models_map_.insert(std::make_pair(contact_status_code, RegressionModel(input_mean_eigen, input_std_eigen, output_mean_eigen, output_std_eigen, dynamics_cost_regression_model)));
    }

}

bool NeuralNetworkInterface::predictFeasibility(std::shared_ptr<ContactState> branching_state)
{
    return true;
}

std::tuple<float, Translation3D, Vector3D> NeuralNetworkInterface::predictDynamicsCost(std::shared_ptr<ContactState> branching_state)
{
    TransformationMatrix feet_mean_transform = branching_state->parent_->getFeetMeanTransform();
    TransformationMatrix inv_feet_mean_transform = inverseTransformationMatrix(feet_mean_transform);

    std::vector< std::shared_ptr<ContactState> > transition_states;

    if(branching_state->prev_move_manip_ == ContactManipulator::R_LEG || branching_state->prev_move_manip_ == ContactManipulator::R_ARM)
    {
        std::shared_ptr<ContactState> mirror_branching_state = branching_state->getMirrorState(feet_mean_transform);
        std::shared_ptr<ContactState> mirror_branching_state_parent = branching_state->parent_->getMirrorState(feet_mean_transform);
        mirror_branching_state->parent_ = mirror_branching_state_parent;
        transition_states.push_back(mirror_branching_state->parent_);
        transition_states.push_back(mirror_branching_state);
    }
    else
    {
        transition_states.push_back(branching_state->parent_);
        transition_states.push_back(branching_state);
    }

    // decide the contact transition code & the poses
    auto transition_code_poses_pair = transition_states[1]->getTransitionCodeAndPoses();
    ContactTransitionCode contact_transition_code = transition_code_poses_pair.first;
    std::vector<RPYTF> contact_manip_pose_vec = transition_code_poses_pair.second;

    std::vector<double> pose_feature_vector(contact_manip_pose_vec.size()*6);
    // std::vector<char> contact_status_vec(int(ContactManipulator::MANIP_NUM)*2);
    int counter = 0;

    for(auto & contact_manip_pose : contact_manip_pose_vec)
    {
        RPYTF transformed_pose = SE3ToXYZRPY(inv_feet_mean_transform * XYZRPYToSE3(contact_manip_pose));

        pose_feature_vector[counter] = transformed_pose.x_;
        pose_feature_vector[counter+1] = transformed_pose.y_;
        pose_feature_vector[counter+2] = transformed_pose.z_;
        pose_feature_vector[counter+3] = transformed_pose.roll_ * DEG2RAD;
        pose_feature_vector[counter+4] = transformed_pose.pitch_ * DEG2RAD;
        pose_feature_vector[counter+5] = transformed_pose.yaw_ * DEG2RAD;

        counter += 6;
    }

    int pose_feature_size = pose_feature_vector.size();
    Eigen::VectorXd feature_vector(pose_feature_size+6);

    for(unsigned int i = 0; i < pose_feature_size; i++)
    {
        feature_vector[i] = pose_feature_vector[i];
    }

    feature_vector.block(pose_feature_size,0,3,1) = (inv_feet_mean_transform * branching_state->parent_->com_.homogeneous()).block(0,0,3,1).cast<double>();
    feature_vector.block(pose_feature_size+3,0,3,1) = (inv_feet_mean_transform.block(0,0,3,3) * branching_state->parent_->com_dot_).cast<double>();

    // std::cout << "feature vector: " << std::endl;

    // for(unsigned int i = 0; i < feature_vector.size(); i++)
    // {
    //     std::cout << feature_vector[i] << ", ";
    // }
    // std::cout << std::endl;

    Eigen::VectorXd prediction = dynamics_cost_regression_models_map_.find(contact_transition_code)->second.predict(feature_vector);

    Translation3D predicted_com = prediction.block(0,0,3,1).cast<float>();
    Translation3D predicted_com_dot = prediction.block(3,0,3,1).cast<float>();

    if(branching_state->prev_move_manip_ == ContactManipulator::R_LEG || branching_state->prev_move_manip_ == ContactManipulator::R_ARM)
    {
        predicted_com[1] = -predicted_com[1];
        predicted_com_dot[1] = -predicted_com_dot[1];
    }

    Translation3D final_com = (feet_mean_transform * predicted_com.homogeneous()).block(0,0,3,1);
    Vector3D final_com_dot = feet_mean_transform.block(0,0,3,3) * predicted_com_dot;
    float dynamics_cost = prediction[6];

    return make_tuple(dynamics_cost, final_com, final_com_dot);
}

bool NeuralNetworkInterface::dynamicsPrediction(std::shared_ptr<ContactState> branching_state, float& dynamics_cost)
{
    bool dynamics_feasibility = predictFeasibility(branching_state);

    if(dynamics_feasibility)
    {
        std::tuple<float, Translation3D, Vector3D> dynamics_prediction = predictDynamicsCost(branching_state);
        dynamics_cost = std::get<0>(dynamics_prediction);
        dynamics_cost = std::max(dynamics_cost,float(0.0));
        branching_state->com_ = std::get<1>(dynamics_prediction);
        branching_state->com_dot_ = std::get<2>(dynamics_prediction);
    }

    return dynamics_feasibility;
}