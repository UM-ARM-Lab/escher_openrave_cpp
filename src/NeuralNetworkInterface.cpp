
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
        // Actually run the image through the model.
        std::vector<tensorflow::Tensor> tf_result;
        tensorflow::Tensor normalized_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, normalized_input.size()}));
        auto normalized_input_eigen_tensor = normalized_input_tensor.matrix<float>();
        for(int i = 0; i < normalized_input.size(); i++)
        {
            normalized_input_eigen_tensor(0,i) = normalized_input[i];
        }
        tensorflow::Status run_status = tf_model_->Run({{"input_1", normalized_input_tensor}}, {"concatenate/concat"}, {}, &tf_result);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            getchar();
        }

        for(int dim_id = 0; dim_id < output_dim_; dim_id++)
        {
            normalized_result[dim_id] = tf_result[0].matrix<float>()(0,dim_id);
        }

        // normalized_result = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >(tf_result[0].flat<double>().data(), tf_result[0].dim_size(0), tf_result[0].dim_size(1));

        //      float,           /* scalar element type */
        //      Eigen::Dynamic,  /* num_rows is a run-time value */
        //      Eigen::Dynamic,  /* num_cols is a run-time value */
        //      Eigen::RowMajor  /* tensorflow::Tensor is always row-major */>>(
        //          t.flat<float>().data(),  /* ptr to data */
        //          t.dim_size(0),           /* num_rows */
        //          t.dim_size(1)            /* num_cols */);

        // // normalized_result = tf_result[0].matrix<float>().block(0,0,output_dim_,1);
        // normalized_result = tf_result[0].matrix<float>();
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

Eigen::MatrixXd RegressionModel::predict(Eigen::MatrixXd input, NeuralNetworkModelType model_type)
{
    size_t data_num = input.cols();
    Eigen::MatrixXd normalized_input = (input - input_mean_.replicate(1,data_num)).cwiseQuotient(input_std_.replicate(1,data_num));
    Eigen::MatrixXd normalized_result(output_dim_, data_num);
    // Eigen::MatrixXf normalized_result_tmp(data_num, output_dim_);

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
            for(int dim_id = 0; dim_id < output_dim_; dim_id++)
            {
                normalized_result(dim_id,data_id) = fdeep_result[data_id][0].get(0,0,0,0,dim_id);
            }
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

        tensorflow::Status run_status = tf_model_->Run({{"input_1", normalized_input_tensor}}, {"concatenate/concat"}, {}, &tf_result);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            getchar();
        }

        for(int dim_id = 0; dim_id < output_dim_; dim_id++)
        {
            for(int data_id = 0; data_id < data_num; data_id++)
            {
                normalized_result(dim_id,data_id) = tf_result[0].matrix<float>()(data_id,dim_id);
            }
        }

        // normalized_result_tmp = Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >(tf_result[0].flat<float>().data(), tf_result[0].dim_size(0), tf_result[0].dim_size(1));
        // normalized_result = normalized_result_tmp.transpose().cast<double>();

        // normalized_result = tf_result[0].matrix<double>();
    }

    Eigen::MatrixXd result = normalized_result.cwiseProduct(output_std_.replicate(1,data_num)) + output_mean_.replicate(1,data_num);

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

        // load tensorflow model
        std::shared_ptr<tensorflow::Session> dynamics_cost_regression_tf_model;
        tensorflow::Status load_regression_graph_status = LoadTensorflowGraph(contact_transition_regression_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".pb", &dynamics_cost_regression_tf_model);
        if (!load_regression_graph_status.ok())
        {
            LOG(ERROR) << load_regression_graph_status;
            getchar();
        }

        auto objective_regression_input_mean_std = readMeanStd(contact_transition_regression_model_file_path + "input_mean_std_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".txt");
        auto objective_regression_output_mean_std = readMeanStd(contact_transition_regression_model_file_path + "output_mean_std_" + std::to_string(contact_status_code_int) + regression_model_parameter_string + ".txt");

        contact_transition_dynamics_cost_regression_models_map_.insert(std::make_pair(contact_status_code, RegressionModel(objective_regression_input_mean_std.first,
                                                                                                                           objective_regression_input_mean_std.second,
                                                                                                                           objective_regression_output_mean_std.first,
                                                                                                                           objective_regression_output_mean_std.second,
                                                                                                                           dynamics_cost_regression_fdeep_model,
                                                                                                                           dynamics_cost_regression_tf_model)));


        // load the classification neural network
        std::string calssification_model_parameter_string = "_0.0001_256_0.1";

        // load frugally-deep model
        std::shared_ptr<fdeep::model> feasibility_calssification_fdeep_model = std::make_shared<fdeep::model>(fdeep::load_model(contact_transition_classification_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + calssification_model_parameter_string + ".json", false, null_logger));

        // load tensorflow model
        std::shared_ptr<tensorflow::Session> feasibility_calssification_tf_model;
        tensorflow::Status load_classification_graph_status = LoadTensorflowGraph(contact_transition_classification_model_file_path + "nn_model_" + std::to_string(contact_status_code_int) + calssification_model_parameter_string + ".pb", &feasibility_calssification_tf_model);
        if (!load_classification_graph_status.ok())
        {
            LOG(ERROR) << load_classification_graph_status;
            getchar();
        }

        auto feasibility_classification_input_mean_std = readMeanStd(contact_transition_classification_model_file_path + "input_mean_std_" + std::to_string(contact_status_code_int) + calssification_model_parameter_string + ".txt");

        contact_transition_feasibility_calssification_models_map_.insert(std::make_pair(contact_status_code, ClassificationModel(feasibility_classification_input_mean_std.first,
                                                                                                                                 feasibility_classification_input_mean_std.second,
                                                                                                                                 feasibility_calssification_fdeep_model,
                                                                                                                                 feasibility_calssification_tf_model)));


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

    tensorflow::SessionOptions option;
    tensorflow::graph::SetDefaultDevice("/CPU:0", &graph_def);
    // tensorflow::graph::SetDefaultDevice("/device:GPU:0", &graph_def);
    // option.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.8);
    // option.config.mutable_gpu_options()->set_allow_growth(true);
    session->reset(tensorflow::NewSession(option));

    // session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
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

std::vector< std::tuple<bool, float, Translation3D, Vector3D> > NeuralNetworkInterface::predictContactTransitionDynamicsCost(std::vector< std::shared_ptr<ContactState> > branching_state_vec, NeuralNetworkModelType model_type)
{
    int states_num = branching_state_vec.size();

    std::vector< std::tuple<bool, float, Translation3D, Vector3D> > contact_transition_dynamics_vec(states_num);
    std::vector<bool> contact_transition_dynamics_feasibility_vec(states_num);
    std::unordered_map<ContactTransitionCode, std::vector<int>, EnumClassHash> contact_transition_code_branching_state_indices_map;
    std::vector<Eigen::VectorXd> feature_vector_vec(states_num);


    if(states_num != 0)
    {
        // construct the feature vectors for each contact transition
        int data_id = 0;
        for(auto branching_state : branching_state_vec)
        {
            if(!branching_state->is_root_)
            {
                std::shared_ptr<ContactState> standard_input_state = branching_state->getStandardInputState(DynOptApplication::CONTACT_TRANSITION_DYNOPT);
                std::shared_ptr<ContactState> prev_state = standard_input_state->parent_;

                // decide the contact transition code & the poses
                auto transition_code_poses_pair = standard_input_state->getTransitionCodeAndPoses();
                ContactTransitionCode contact_transition_code = transition_code_poses_pair.first;
                std::vector<RPYTF> contact_manip_pose_vec = transition_code_poses_pair.second;

                if(contact_transition_code_branching_state_indices_map.find(contact_transition_code) == contact_transition_code_branching_state_indices_map.end())
                {
                    contact_transition_code_branching_state_indices_map[contact_transition_code] = {data_id};
                }
                else
                {
                    contact_transition_code_branching_state_indices_map[contact_transition_code].push_back(data_id);
                }

                feature_vector_vec[data_id] = constructFeatureVector(contact_manip_pose_vec, prev_state->com_, prev_state->com_dot_);
            }

            data_id++;
        }


        for(auto & contact_transition_code_branching_state_index_pair : contact_transition_code_branching_state_indices_map)
        {
            // construct the feasibility feature matrix to query the contact transition feasibility network
            ContactTransitionCode contact_transition_code = contact_transition_code_branching_state_index_pair.first;
            int input_dim = feature_vector_vec[contact_transition_code_branching_state_index_pair.second[0]].size();
            int data_num = contact_transition_code_branching_state_index_pair.second.size();

            Eigen::MatrixXd feature_matrix(input_dim, data_num);

            int query_data_id = 0;
            for(auto & data_id : contact_transition_code_branching_state_index_pair.second)
            {
                feature_matrix.col(query_data_id) = feature_vector_vec[data_id];
                query_data_id++;
            }

            std::vector<float> contact_transition_dynamics_feasibility_predictions = contact_transition_feasibility_calssification_models_map_.find(contact_transition_code)->second.predict(feature_matrix, model_type);
            Eigen::MatrixXd contact_transition_dynamics_objective_predictions = contact_transition_dynamics_cost_regression_models_map_.find(contact_transition_code)->second.predict(feature_matrix, model_type);

            // construct the output vector
            query_data_id = 0;
            for(auto & data_id : contact_transition_code_branching_state_index_pair.second)
            {
                if(contact_transition_dynamics_feasibility_predictions[query_data_id] >= 0.5)
                // if(true)
                {
                    Translation3D predicted_com = contact_transition_dynamics_objective_predictions.block(0,query_data_id,3,1).cast<float>();
                    Vector3D predicted_com_dot = contact_transition_dynamics_objective_predictions.block(3,query_data_id,3,1).cast<float>();

                    if(branching_state_vec[data_id]->prev_move_manip_ == ContactManipulator::R_LEG ||
                       branching_state_vec[data_id]->prev_move_manip_ == ContactManipulator::R_ARM)
                    {
                        predicted_com[1] = -predicted_com[1];
                        predicted_com_dot[1] = -predicted_com_dot[1];
                    }

                    TransformationMatrix feet_mean_transform = branching_state_vec[data_id]->parent_->getFeetMeanTransform();

                    Translation3D final_com = (feet_mean_transform * predicted_com.homogeneous()).block(0,0,3,1);
                    Vector3D final_com_dot = feet_mean_transform.block(0,0,3,3) * predicted_com_dot;
                    float dynamics_cost = contact_transition_dynamics_objective_predictions(6,query_data_id);

                    contact_transition_dynamics_vec[data_id] = make_tuple(true, dynamics_cost, final_com, final_com_dot);
                }
                else
                {
                    Translation3D dummy_com(0,0,0);
                    Vector3D dummy_com_dot(0,0,0);
                    float dummy_dynamics_cost = 0;
                    contact_transition_dynamics_vec[data_id] = make_tuple(false, dummy_dynamics_cost, dummy_com, dummy_com_dot);
                }
                query_data_id++;
            }
        }
    }

    return contact_transition_dynamics_vec;
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

    float dynamics_feasibility_prediction = contact_transition_feasibility_calssification_models_map_.find(contact_transition_code)->second.predict(feature_vector, model_type);
    dynamics_feasibility = (dynamics_feasibility_prediction >= 0.5);

    // if(contact_transition_code == ContactTransitionCode::FEET_AND_ONE_HAND_BREAK_HAND || contact_transition_code == ContactTransitionCode::FEET_AND_TWO_HANDS_BREAK_HAND)
    // {
    //     dynamics_feasibility = true;
    // }

    // dynamics_feasibility = true;

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

std::vector<bool> NeuralNetworkInterface::predictContactTransitionDynamics(std::vector< std::shared_ptr<ContactState> > branching_state_vec, std::vector<float>& dynamics_cost_vec, NeuralNetworkModelType model_type)
{
    int state_num = branching_state_vec.size();
    dynamics_cost_vec.resize(state_num);
    std::vector<bool> dynamics_feasibility_vec(state_num);

    std::vector< std::tuple<bool, float, Translation3D, Vector3D> > dynamics_prediction_vec = predictContactTransitionDynamicsCost(branching_state_vec, model_type);

    for(int data_id; data_id < state_num; data_id++)
    {
        if(branching_state_vec[data_id]->is_root_)
        {
            dynamics_feasibility_vec[data_id] = true;
            dynamics_cost_vec[data_id] = 0;
        }
        else
        {
            dynamics_feasibility_vec[data_id] = std::get<0>(dynamics_prediction_vec[data_id]);
            dynamics_cost_vec[data_id] = std::get<1>(dynamics_prediction_vec[data_id]);
            dynamics_cost_vec[data_id] = std::max(dynamics_cost_vec[data_id],float(0.0));
            branching_state_vec[data_id]->com_ = std::get<2>(dynamics_prediction_vec[data_id]);
            branching_state_vec[data_id]->com_dot_ = std::get<3>(dynamics_prediction_vec[data_id]);
        }
    }

    return dynamics_feasibility_vec;
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

    std::vector<ContactManipulator> mirror_manip_vec = {ContactManipulator::R_LEG, ContactManipulator::L_LEG, ContactManipulator::R_ARM, ContactManipulator::L_ARM};

    // RotationMatrix mirror_matrix;
    // mirror_matrix << 1,  0, 0,
    //                  0, -1, 0,
    //                  0,  0, 1;

    if(one_step_capture_state_vec.size() != 0)
    {
        auto time_start_predict_one_step_capture_dynamics = std::chrono::high_resolution_clock::now();

        // custom build the query vector vector
        std::unordered_map< std::pair<std::size_t,int>, std::shared_ptr<ContactState>, PairHash> prev_standard_states;
        int data_id = 0;
        for(auto one_step_capture_state : one_step_capture_state_vec)
        {
            std::shared_ptr<ContactState> prev_state = one_step_capture_state->parent_;
            ContactManipulator move_manip = one_step_capture_state->prev_move_manip_;
            TransformationMatrix reference_frame = one_step_capture_state->parent_->getFeetMeanTransform();
            TransformationMatrix inv_reference_frame = inverseTransformationMatrix(reference_frame);

            int manip_side = (move_manip == ContactManipulator::R_LEG || move_manip == ContactManipulator::R_ARM) ? 1 : 0;
            std::size_t zero_step_capture_state_hash = std::hash<ContactState>()(*(one_step_capture_state->parent_));
            std::pair<std::size_t, int> prev_state_key = std::make_pair(zero_step_capture_state_hash, manip_side);
            if(prev_standard_states.find(prev_state_key) == prev_standard_states.end())
            {
                if(move_manip == ContactManipulator::R_LEG || move_manip == ContactManipulator::R_ARM)
                {
                    prev_state = one_step_capture_state->parent_->getMirrorState(reference_frame);
                }
                prev_state = prev_state->getCenteredState(reference_frame);
                prev_standard_states[prev_state_key] = prev_state;
            }
            else
            {
                prev_state = prev_standard_states[prev_state_key];
            }

            // get reference frame
            ContactManipulator standard_move_manip = move_manip;
            std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status = prev_state->stances_vector_[0]->ee_contact_status_;
            std:array<RPYTF,ContactManipulator::MANIP_NUM> ee_contact_poses = prev_state->stances_vector_[0]->ee_contact_poses_;
            TransformationMatrix capture_pose = inv_reference_frame * XYZRPYToSE3(one_step_capture_state->stances_vector_[0]->ee_contact_poses_[move_manip]);

            if(move_manip == ContactManipulator::R_LEG || move_manip == ContactManipulator::R_ARM)
            {
                standard_move_manip = mirror_manip_vec[int(move_manip)];
                // capture_pose.block(0,0,3,3) = mirror_matrix * capture_pose.block(0,0,3,3) * mirror_matrix;
                capture_pose(0,1) = -capture_pose(0,1); // mirror
                capture_pose(1,0) = -capture_pose(1,0);
                capture_pose(2,1) = -capture_pose(2,1);
                capture_pose(1,2) = -capture_pose(1,2);
                capture_pose(1,3) = -capture_pose(1,3);
            }

            ee_contact_poses[int(standard_move_manip)] = SE3ToXYZRPY(capture_pose);
            std::vector<ContactManipulator> contact_manipulators;

            OneStepCaptureCode one_step_capture_code;
            if(ee_contact_status[ContactManipulator::R_LEG] && standard_move_manip == ContactManipulator::L_LEG)
            {
                one_step_capture_code = OneStepCaptureCode::ONE_FOOT_ADD_FOOT;
                contact_manipulators = {ContactManipulator::R_LEG, ContactManipulator::L_LEG};
            }
            else if(ee_contact_status[ContactManipulator::R_LEG] && standard_move_manip == ContactManipulator::L_ARM)
            {
                one_step_capture_code = OneStepCaptureCode::ONE_FOOT_ADD_OUTER_HAND;
                contact_manipulators = {ContactManipulator::R_LEG, ContactManipulator::L_ARM};
            }
            else if(ee_contact_status[ContactManipulator::L_LEG] && standard_move_manip == ContactManipulator::L_ARM)
            {
                one_step_capture_code = OneStepCaptureCode::ONE_FOOT_ADD_INNER_HAND;
                contact_manipulators = {ContactManipulator::L_LEG, ContactManipulator::L_ARM};
            }

            Eigen::VectorXd feature_vector(contact_manipulators.size()*6+6);

            unsigned int counter = 0;
            for(auto & contact_manip : contact_manipulators)
            {
                feature_vector[counter]   = ee_contact_poses[contact_manip].x_;
                feature_vector[counter+1] = ee_contact_poses[contact_manip].y_;
                feature_vector[counter+2] = ee_contact_poses[contact_manip].z_;
                feature_vector[counter+3] = ee_contact_poses[contact_manip].roll_ * DEG2RAD;
                feature_vector[counter+4] = ee_contact_poses[contact_manip].pitch_ * DEG2RAD;
                feature_vector[counter+5] = ee_contact_poses[contact_manip].yaw_ * DEG2RAD;

                counter += 6;
            }
            feature_vector.block(feature_vector.size()-6, 0, 3, 1) = prev_state->com_.cast<double>();
            feature_vector.block(feature_vector.size()-3, 0, 3, 1) = prev_state->lmom_.cast<double>();
            feature_vector_vec[data_id] = feature_vector;

            if(capture_code_one_step_capture_state_indices_map.find(one_step_capture_code) == capture_code_one_step_capture_state_indices_map.end())
            {
                capture_code_one_step_capture_state_indices_map[one_step_capture_code] = {data_id};
                capture_code_one_step_capture_state_indices_map[one_step_capture_code].reserve(one_step_capture_state_vec.size());
            }
            else
            {
                capture_code_one_step_capture_state_indices_map[one_step_capture_code].push_back(data_id);
            }

            data_id++;
        }

        // // structured but slow way to build the query vector vector
        // int data_id = 0;
        // for(auto one_step_capture_state : one_step_capture_state_vec)
        // {
        //     // get reference frame
        //     std::shared_ptr<ContactState> standard_input_state = one_step_capture_state->getStandardInputState(DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);
        //     std::shared_ptr<ContactState> prev_state = standard_input_state->parent_;

        //     // decide the motion code & the poses
        //     auto motion_code_poses_pair = standard_input_state->getOneStepCapturabilityCodeAndPoses();
        //     OneStepCaptureCode one_step_capture_code = motion_code_poses_pair.first;
        //     std::vector<RPYTF> contact_manip_pose_vec = motion_code_poses_pair.second;

        //     if(capture_code_one_step_capture_state_indices_map.find(one_step_capture_code) == capture_code_one_step_capture_state_indices_map.end())
        //     {
        //         capture_code_one_step_capture_state_indices_map[one_step_capture_code] = {data_id};
        //     }
        //     else
        //     {
        //         capture_code_one_step_capture_state_indices_map[one_step_capture_code].push_back(data_id);
        //     }

        //     feature_vector_vec[data_id] = constructFeatureVector(contact_manip_pose_vec, prev_state->com_, prev_state->lmom_);

        //     data_id++;
        // }

        auto time_after_collect_feature_vectors = std::chrono::high_resolution_clock::now();

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

            // std::cout << data_num << " ";
        }

        auto time_after_query_one_step_capturabiltiy_classifiers = std::chrono::high_resolution_clock::now();

        // std::cout << std::endl;
        // std::cout << "  get feature vector time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_collect_feature_vectors - time_start_predict_one_step_capture_dynamics).count() << std::endl;
        // std::cout << "  query classifier time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_query_one_step_capturabiltiy_classifiers - time_after_collect_feature_vectors).count() << std::endl;
        // std::cout << data_id << std::endl;
        // std::cout << prev_standard_states.size() << std::endl;
        // // getchar();
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