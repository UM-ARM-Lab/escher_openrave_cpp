#ifndef NEURALNETWORKINTERFACE_HPP
#define NEURALNETWORKINTERFACE_HPP

#include <fdeep/fdeep.hpp>

class ClassificationModel
{
    public:
        ClassificationModel(Eigen::VectorXd _input_mean, Eigen::VectorXd _input_std, fdeep::model _model):
        input_mean_(_input_mean),
        input_std_(_input_std),
        input_dim_(_input_mean.size()),
        model_(_model) {};
        int predict(Eigen::VectorXd input);

    private:
        const int input_dim_;

        Eigen::VectorXd input_mean_;
        Eigen::VectorXd input_std_;

        fdeep::model model_;
};

class RegressionModel
{
    public:
        RegressionModel(Eigen::VectorXd _input_mean, Eigen::VectorXd _input_std, Eigen::VectorXd _output_mean, Eigen::VectorXd _output_std, fdeep::model _model):
        input_mean_(_input_mean),
        input_std_(_input_std),
        output_mean_(_output_mean),
        output_std_(_output_std),
        input_dim_(_input_mean.size()),
        output_dim_(_output_mean.size()),
        model_(_model) {};
        Eigen::VectorXd predict(Eigen::VectorXd input);

    private:
        const int input_dim_;
        const int output_dim_;

        Eigen::VectorXd input_mean_;
        Eigen::VectorXd input_std_;
        Eigen::VectorXd output_mean_;
        Eigen::VectorXd output_std_;

        fdeep::model model_;
};

class NeuralNetworkInterface
{
    public:
        NeuralNetworkInterface(std::string model_file_path);
        ~NeuralNetworkInterface(){};

        bool predictFeasibility(std::shared_ptr<ContactState> branching_state);
        std::tuple<float, Translation3D, Vector3D> predictDynamicsCost(std::shared_ptr<ContactState> branching_state);

    private:
        std::unordered_map<std::string, ClassificationModel> feasibility_calssification_models_map_;
        std::unordered_map<std::string, RegressionModel> dynamics_cost_regression_models_map_;


};

#endif