
#include "Utilities.hpp"

int ClassificationModel::predict(Eigen::VectorXd input)
{
    Eigen::VectorXd normalized_input = (input - input_mean_).cwiseQuotient(input_std_);

    fdeep::float_vec input_vec(input.data(), input.data() + input.size());
    fdeep::shared_float_vec input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(input_vec));

    fdeep::tensor3s result = model_.predict({fdeep::tensor3(fdeep::shape3(input_dim_, 1, 1), input_vec_ref)});

    return result[0].get(0,0,0);
}

Eigen::VectorXd RegressionModel::predict(Eigen::VectorXd input)
{
    Eigen::VectorXd normalized_input = (input - input_mean_).cwiseQuotient(input_std_);

    fdeep::float_vec input_vec(input.data(), input.data() + input.size());
    fdeep::shared_float_vec input_vec_ref = fplus::make_shared_ref<fdeep::float_vec>(std::move(input_vec));

    auto result = model_.predict({fdeep::tensor3(fdeep::shape3(input_dim_, 1, 1), input_vec_ref)});

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
        fdeep::model feasibility_calssification_model = fdeep::load_model(model_file_path + "feasibility_calssification_64_64_" + contact_status_code + ".json");
        fdeep::model dynamics_cost_regression_model = fdeep::load_model(model_file_path + "dynamics_cost_regression_64_64_" + contact_status_code + ".json");

        feasibility_calssification_models_map_.insert(std::make_pair(contact_status_code, ClassificationModel(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), feasibility_calssification_model)));
        dynamics_cost_regression_models_map_.insert(std::make_pair(contact_status_code, RegressionModel(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), dynamics_cost_regression_model)));
    }

}

bool NeuralNetworkInterface::predictFeasibility(std::shared_ptr<ContactState> branching_state)
{

}

std::tuple<float, Translation3D, Vector3D> NeuralNetworkInterface::predictDynamicsCost(std::shared_ptr<ContactState> branching_state)
{
    std::vector<char> contact_status_vec(8);
    int counter = 0;
    for(auto & ee_contact_status : branching_state->parent_->stances_vector_[0]->ee_contact_status_)
    {
        if(ee_contact_status)
        {
            contact_status_vec[counter] = '1';
        }
        else
        {
            contact_status_vec[counter] = '0';
        }
        counter++;
    }

    for(auto & ee_contact_status : branching_state->stances_vector_[0]->ee_contact_status_)
    {
        if(ee_contact_status)
        {
            contact_status_vec[counter] = '1';
        }
        else
        {
            contact_status_vec[counter] = '0';
        }
        counter++;
    }

    std::string contact_status_code(contact_status_vec.begin(), contact_status_vec.end());

    // dynamics_cost_regression_models_map_[contact_status_code].predict();


}