#ifndef NEURALNETWORKINTERFACE_HPP
#define NEURALNETWORKINTERFACE_HPP

// #define FDEEP_FLOAT_TYPE double

class ClassificationModel
{
    public:
        ClassificationModel(Eigen::VectorXd _input_mean, Eigen::VectorXd _input_std, std::shared_ptr<fdeep::model> _fdeep_model=nullptr, std::shared_ptr<tensorflow::Session> _tf_model=nullptr):
        input_mean_(_input_mean),
        input_std_(_input_std),
        input_dim_(_input_mean.size()),
        fdeep_model_(_fdeep_model),
        tf_model_(_tf_model) {};

        float predict(Eigen::VectorXd input, NeuralNetworkModelType model_type);
        std::vector<float> predict(Eigen::MatrixXd input, NeuralNetworkModelType model_type);

    private:
        const int input_dim_;

        Eigen::VectorXd input_mean_;
        Eigen::VectorXd input_std_;

        std::shared_ptr<fdeep::model> fdeep_model_;
        std::shared_ptr<tensorflow::Session> tf_model_;
};

class RegressionModel
{
    public:
        RegressionModel(Eigen::VectorXd _input_mean, Eigen::VectorXd _input_std, Eigen::VectorXd _output_mean, Eigen::VectorXd _output_std, std::shared_ptr<fdeep::model> _fdeep_model=nullptr, std::shared_ptr<tensorflow::Session> _tf_model=nullptr):
        input_mean_(_input_mean),
        input_std_(_input_std),
        output_mean_(_output_mean),
        output_std_(_output_std),
        input_dim_(_input_mean.size()),
        output_dim_(_output_mean.size()),
        fdeep_model_(_fdeep_model),
        tf_model_(_tf_model) {};

        Eigen::VectorXd predict(Eigen::VectorXd input, NeuralNetworkModelType model_type);
        Eigen::MatrixXd predict(Eigen::MatrixXd input, NeuralNetworkModelType model_type);

    private:
        const int input_dim_;
        const int output_dim_;

        Eigen::VectorXd input_mean_;
        Eigen::VectorXd input_std_;
        Eigen::VectorXd output_mean_;
        Eigen::VectorXd output_std_;

        std::shared_ptr<fdeep::model> fdeep_model_;
        std::shared_ptr<tensorflow::Session> tf_model_;
};

class NeuralNetworkInterface
{
    public:
        NeuralNetworkInterface(std::string contact_transition_regression_model_file_path,
                               std::string contact_transition_classification_model_file_path,
                               std::string zero_step_capturability_classification_model_file_path,
                               std::string one_step_capturability_classification_model_file_path);

        // bool predictFeasibility(std::shared_ptr<ContactState> branching_state);
        std::tuple<bool, float, Translation3D, Vector3D> predictContactTransitionDynamicsCost(std::shared_ptr<ContactState> branching_state, NeuralNetworkModelType model_type);
        std::vector< std::tuple<bool, float, Translation3D, Vector3D> > predictContactTransitionDynamicsCost(std::vector< std::shared_ptr<ContactState> > branching_state_vec, NeuralNetworkModelType model_type);
        bool predictContactTransitionDynamics(std::shared_ptr<ContactState> branching_state, float& dynamics_cost, NeuralNetworkModelType model_type);
        std::vector<bool> predictContactTransitionDynamics(std::vector< std::shared_ptr<ContactState> > branching_state_vec, std::vector<float>& dynamics_cost_vec, NeuralNetworkModelType model_type);
        bool predictZeroStepCaptureDynamics(std::shared_ptr<ContactState> zero_step_capture_state, NeuralNetworkModelType model_type);
        std::vector<bool> predictZeroStepCaptureDynamics(std::vector< std::shared_ptr<ContactState> > zero_step_capture_state_vec, NeuralNetworkModelType model_type);
        bool predictOneStepCaptureDynamics(std::shared_ptr<ContactState> one_step_capture_state, NeuralNetworkModelType model_type);
        std::vector<bool> predictOneStepCaptureDynamics(std::vector< std::shared_ptr<ContactState> > one_step_capture_state_vec, NeuralNetworkModelType model_type);

        Eigen::VectorXd getOneStepCaptureFeatureVector(std::shared_ptr<ContactState> one_step_capture_state);

    private:
        std::unordered_map<ContactTransitionCode, ClassificationModel, EnumClassHash> contact_transition_feasibility_calssification_models_map_;
        std::unordered_map<ContactTransitionCode, RegressionModel, EnumClassHash> contact_transition_dynamics_cost_regression_models_map_;
        std::unordered_map<ZeroStepCaptureCode, ClassificationModel, EnumClassHash> zero_step_capturability_calssification_models_map_;
        std::unordered_map<OneStepCaptureCode, ClassificationModel, EnumClassHash> one_step_capturability_calssification_models_map_;

        std::pair<Eigen::VectorXd, Eigen::VectorXd> readMeanStd(std::string file_path);
        Eigen::VectorXd constructFeatureVector(std::vector<RPYTF>& contact_manip_pose_vec, Translation3D& com, Vector3D& com_dot);

        tensorflow::Status LoadTensorflowGraph(const std::string graph_file_name, std::shared_ptr<tensorflow::Session>* session);


};

#endif