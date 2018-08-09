
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

    fdeep::float_vec input_vec(input.data(), input.data() + input.size());
    fdeep::shared_float_vec input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(input_vec));

    auto result = model_->predict({fdeep::tensor3(fdeep::shape3(input_dim_, 1, 1), input_vec_ref)});

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
    std::vector<std::string> state_transition_contact_status_code_vec = {"11001100"};

    for(auto & contact_status_code : state_transition_contact_status_code_vec)
    {
        // load the neural network
        // std::shared_ptr<fdeep::model> feasibility_calssification_model = std::make_shared<fdeep::model>(fdeep::load_model(model_file_path + "feasibility_calssification_64_64_" + contact_status_code + ".json"));
        std::shared_ptr<fdeep::model> dynamics_cost_regression_model = std::make_shared<fdeep::model>(fdeep::load_model(model_file_path + "nn_model_" + contact_status_code + "_0.0005_192_0.0.json"));

        // load the input/output mean and std
        std::ifstream f_input_mean_std, f_output_mean_std;

        f_input_mean_std.open(model_file_path + "input_mean_std_" + contact_status_code + "_0.0005_192_0.0.txt",std::ifstream::in);
        f_output_mean_std.open(model_file_path + "output_mean_std_" + contact_status_code + "_0.0005_192_0.0.txt",std::ifstream::in);

        std::vector<double> input_mean, input_std, output_mean, output_std;
        double mean_data, std_data;

        while(!f_input_mean_std.eof())
        {
            f_input_mean_std >> mean_data;
            input_mean.push_back(mean_data);
            f_input_mean_std >> std_data;
            input_std.push_back(std_data);
        }
        while(!f_output_mean_std.eof())
        {
            f_output_mean_std >> mean_data;
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
    std::vector<double> pose_feature_vector;
    std::vector<char> contact_status_vec(int(ContactManipulator::MANIP_NUM)*2);
    int counter = 0;
    std::vector< std::shared_ptr<ContactState> > transition_states = {branching_state->parent_, branching_state};

    TransformationMatrix inv_feet_mean_transform = inverseTransformationMatrix(branching_state->parent_->getFeetMeanTransform());

    for(auto & state_ptr : transition_states)
    {
        std::shared_ptr<Stance> stance_ptr = state_ptr->stances_vector_[0];

        for(int i = 0; i < ContactManipulator::MANIP_NUM; i++)
        {
            if(stance_ptr->ee_contact_status_[i])
            {
                contact_status_vec[counter] = '1';

                RPYTF transformed_pose = SE3ToXYZRPY(inv_feet_mean_transform * XYZRPYToSE3(stance_ptr->ee_contact_poses_[i]));

                pose_feature_vector.push_back(transformed_pose.x_);
                pose_feature_vector.push_back(transformed_pose.y_);
                pose_feature_vector.push_back(transformed_pose.z_);
                pose_feature_vector.push_back(transformed_pose.roll_ * DEG2RAD);
                pose_feature_vector.push_back(transformed_pose.pitch_ * DEG2RAD);
                pose_feature_vector.push_back(transformed_pose.yaw_ * DEG2RAD);
            }
            else
            {
                contact_status_vec[counter] = '0';
            }
            counter++;
        }
    }

    int pose_feature_size = pose_feature_vector.size();
    Eigen::VectorXd feature_vector(pose_feature_size+6);

    for(int i = 0; i < pose_feature_size; i++)
    {
        feature_vector[i] = pose_feature_vector[i];
    }

    feature_vector.block(pose_feature_size,0,3,1) = (inv_feet_mean_transform * branching_state->parent_->com_.homogeneous()).block(0,0,3,1).cast<double>();
    feature_vector.block(pose_feature_size+3,0,3,1) = (inv_feet_mean_transform.block(0,0,3,3) * branching_state->parent_->com_dot_).cast<double>();

    std::string contact_status_code(contact_status_vec.begin(), contact_status_vec.end());

    Eigen::VectorXd prediction = dynamics_cost_regression_models_map_.find(contact_status_code)->second.predict(feature_vector);

    Translation3D final_com = prediction.block(0,0,3,1).cast<float>();
    Vector3D final_com_dot = prediction.block(3,0,3,1).cast<float>();
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
        branching_state->com_ = std::get<1>(dynamics_prediction);
        branching_state->com_dot_ = std::get<2>(dynamics_prediction);
    }

    return dynamics_feasibility;
}