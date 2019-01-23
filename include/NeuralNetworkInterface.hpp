#ifndef NEURALNETWORKINTERFACE_HPP
#define NEURALNETWORKINTERFACE_HPP

// #define FDEEP_FLOAT_TYPE double

class ClassificationModel
{
    public:
        ClassificationModel(Eigen::VectorXd _input_mean, Eigen::VectorXd _input_std, std::shared_ptr<fdeep::model> _model):
        input_mean_(_input_mean),
        input_std_(_input_std),
        input_dim_(_input_mean.size()),
        model_(_model) {};
        float predict(Eigen::VectorXd input);

    private:
        const int input_dim_;

        Eigen::VectorXd input_mean_;
        Eigen::VectorXd input_std_;

        std::shared_ptr<fdeep::model> model_;
};

class RegressionModel
{
    public:
        RegressionModel(Eigen::VectorXd _input_mean, Eigen::VectorXd _input_std, Eigen::VectorXd _output_mean, Eigen::VectorXd _output_std, std::shared_ptr<fdeep::model> _model):
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

        std::shared_ptr<fdeep::model> model_;
};

class NeuralNetworkInterface
{
    public:
        NeuralNetworkInterface(std::string contact_transition_regression_model_file_path,
                               std::string contact_transition_classification_model_file_path,
                               std::string zero_step_capturability_classification_model_file_path,
                               std::string one_step_capturability_classification_model_file_path);

        // bool predictFeasibility(std::shared_ptr<ContactState> branching_state);
        std::tuple<bool, float, Translation3D, Vector3D> predictDynamicsCost(std::shared_ptr<ContactState> branching_state);
        bool dynamicsPrediction(std::shared_ptr<ContactState> branching_state, float& dynamics_cost);

    private:
        std::unordered_map<ContactTransitionCode, ClassificationModel, EnumClassHash> contact_transition_feasibility_calssification_models_map_;
        std::unordered_map<ContactTransitionCode, RegressionModel, EnumClassHash> contact_transition_dynamics_cost_regression_models_map_;
        std::unordered_map<ZeroStepCaptureCode, ClassificationModel, EnumClassHash> zero_step_capturability_calssification_models_map_;
        std::unordered_map<OneStepCaptureCode, ClassificationModel, EnumClassHash> one_step_capturability_calssification_models_map_;

        std::pair<Eigen::VectorXd, Eigen::VectorXd> readMeanStd(std::string file_path);


};

#endif