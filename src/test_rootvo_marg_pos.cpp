#include <iostream>
#include <random>

#include <include/ba_solver/root_vio/problem_vo.hpp>

using Scalar = float;      // 方便切换浮点数精度

/*
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
    Frame(rootVIO::Matrix3<Scalar> R, rootVIO::Vector3<Scalar> t) : Rwc(R), qwc(R), twc(t) {};
    rootVIO::Matrix3<Scalar> Rwc;
    rootVIO::Quaternion<Scalar> qwc;
    rootVIO::Vector3<Scalar> twc;
    std::unordered_map<int, rootVIO::Vector3<Scalar>> featurePerId; // 该帧观测到的特征以及特征id
};


/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(std::vector<Frame> &cameraPoses, std::vector<rootVIO::Vector3<Scalar>> &points) {
    int featureNums = 5;   // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 5;      // 相机数目

    double radius = 8;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 16); // 1/16 圆弧
        // 绕 z 轴 旋转
        rootVIO::Matrix3<Scalar> R;
        R = Eigen::AngleAxis<Scalar>(theta, rootVIO::Vector3<Scalar>::UnitZ());
        rootVIO::Vector3<Scalar> t = rootVIO::Vector3<Scalar>(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        cameraPoses.push_back(Frame(R, t));
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
    for (int j = 0; j < featureNums; ++j) {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);

        rootVIO::Vector3<Scalar> Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        points.push_back(Pw);

        // 在每一帧上的观测量
        for (int i = 0; i < poseNums; ++i) {
            rootVIO::Vector3<Scalar> Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            if (Pc.z() < 1e-10) {
                continue;
            }
            Pc = Pc / Pc.z();  // 归一化图像平面
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            cameraPoses[i].featurePerId.insert(std::make_pair(j, Pc));
        }
    }
}





/* 程序主函数入口 */
int main() {
    std::cout << "Test rootVIO solver on Mono BA problem." << std::endl;

    // 第一步：准备测试数据
    std::cout << "\nStep 1: Prepare dataset." << std::endl;
    std::vector<Frame> cameras;
    std::vector<rootVIO::Vector3<Scalar>> points;
    GetSimDataInWordFrame(cameras, points);

    // 第二步：构造 BA 问题求解器，并添加相机与 IMU 外参节点
    std::cout << "\nStep 2: Construct problem solver." << std::endl;
    rootVIO::ProblemVO<Scalar> problem;
    problem.Reset();
    size_t priorSize = 0;
    rootVIO::Quaternion<Scalar> q_bc;
    q_bc.setIdentity();
    rootVIO::Vector3<Scalar> t_bc = rootVIO::Vector3<Scalar>::Zero();
    std::shared_ptr<rootVIO::VertexExPose<Scalar>> exPose(new rootVIO::VertexExPose<Scalar>(q_bc, t_bc));
    problem.AddExPose(exPose);
    priorSize += 6;
    std::cout << "problem has 1 ex pose." << std::endl;

    // 第三步：构造相机 pose 节点，并添加到 problem 中
    std::cout << "\nStep 3: Add camera pose vertices." << std::endl;
    std::default_random_engine generator;
    std::normal_distribution<double> camera_rotation_noise(0., 0.01);
    std::normal_distribution<double> camera_position_noise(0., 0.01);
    std::vector<std::shared_ptr<rootVIO::VertexCameraPose<Scalar>>> cameraVertices;
    for (size_t i = 0; i < cameras.size() - 1; ++i) {
        // 给相机位置和姿态初值增加噪声
        rootVIO::Matrix3<Scalar> noise_R, noise_X, noise_Y, noise_Z;
        noise_X = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitX());
        noise_Y = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitY());
        noise_Z = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitZ());
        noise_R = noise_X * noise_Y * noise_Z;
        rootVIO::Vector3<Scalar> noise_t(camera_position_noise(generator), camera_position_noise(generator), camera_position_noise(generator));
        rootVIO::Quaternion<Scalar> noise_q_wc(noise_R);
        std::shared_ptr<rootVIO::VertexCameraPose<Scalar>> cameraVertex(
            new rootVIO::VertexCameraPose<Scalar>(cameras[i].qwc * noise_q_wc, cameras[i].twc + noise_t));
        cameraVertices.emplace_back(cameraVertex);
        problem.AddCamera(cameraVertex);
        priorSize += 6;
    }
    std::cout << "problem has " << problem.GetCamerasNum() << " cameras." << std::endl;

    // 第四步：构造特征点 position 节点，并添加到 problem 中
    std::cout << "\nStep 4: Add landmark position vertices." << std::endl;
    std::list<std::shared_ptr<rootVIO::VertexLandmark<Scalar>>> landmarkVertices;
    for (size_t i = 0; i < points.size(); ++i) {
        // 为初值添加随机噪声
        std::normal_distribution<double> landmark_position_noise(0., 0.01);
        rootVIO::Vector3<Scalar> noise(landmark_position_noise(generator), landmark_position_noise(generator), landmark_position_noise(generator));
        std::shared_ptr<rootVIO::VertexLandmark<Scalar>> landmarkVertex(new rootVIO::VertexLandmark<Scalar>(points[i] + noise));
        // 构造观测
        std::unordered_map<size_t, std::shared_ptr<rootVIO::CameraObserve<Scalar>>> observes;
        for (size_t j = 0; j < cameras.size() - 1; ++j) {
            std::shared_ptr<rootVIO::HuberKernel<Scalar>> kernel(new rootVIO::HuberKernel<Scalar>(1.0));
            std::shared_ptr<rootVIO::CameraObserve<Scalar>> observe(new rootVIO::CameraObserve<Scalar>(
                cameraVertices[j], cameras[j].featurePerId.find(i)->second.head<2>(), kernel));
            observes.insert(std::make_pair(j, observe));
        }
        
        landmarkVertices.emplace_back(landmarkVertex);
        problem.AddLandmark(landmarkVertex, observes, 3);
    }
    std::cout << "problem has " << problem.GetLandmarksNum() << " landmarks." << std::endl;

    // 第五步：设置求解器的相关参数，开始边缘化最旧帧
    std::cout << "\nStep 5: Start marginalize oldest frame." << std::endl;
    problem.Marginalize(0, priorSize - 6);
    rootVIO::MatrixX<Scalar> prior_J;
    rootVIO::VectorX<Scalar> prior_r;
    problem.GetPrior(prior_J, prior_r);
    std::cout << "  prior J is\n" << prior_J << std::endl << "  prior r is\n" << prior_r << std::endl;
    std::cout << "  prior H is\n" << prior_J.transpose() * prior_J << std::endl;
    std::cout << "  prior b is\n" << - prior_J.transpose() * prior_r << std::endl;

    // 第六步：构造新的边缘化问题
    std::cout << "\nStep 6: Construct problem2 solver." << std::endl;
    rootVIO::ProblemVO<Scalar> problem2;
    problem2.Reset();
    priorSize = 6;
    problem2.AddExPose(exPose);

    // 第七步：构造相机 pose 节点，并添加到 problem2 中
    std::cout << "\nStep 7: Add camera pose vertices." << std::endl;
    cameraVertices.clear();
    for (size_t i = 1; i < cameras.size(); ++i) {
        // 给相机位置和姿态初值增加噪声
        rootVIO::Matrix3<Scalar> noise_R, noise_X, noise_Y, noise_Z;
        noise_X = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitX());
        noise_Y = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitY());
        noise_Z = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitZ());
        noise_R = noise_X * noise_Y * noise_Z;
        rootVIO::Vector3<Scalar> noise_t(camera_position_noise(generator), camera_position_noise(generator), camera_position_noise(generator));
        rootVIO::Quaternion<Scalar> noise_q_wc(noise_R);
        std::shared_ptr<rootVIO::VertexCameraPose<Scalar>> cameraVertex(
            new rootVIO::VertexCameraPose<Scalar>(cameras[i].qwc * noise_q_wc, cameras[i].twc + noise_t));
        cameraVertices.emplace_back(cameraVertex);
        problem2.AddCamera(cameraVertex);
        priorSize += 6;
    }
    std::cout << "problem2 has " << problem2.GetCamerasNum() << " cameras." << std::endl;

    // 第八步：为 problem2 添加先验信息
    std::cout << "\nStep 8: Add prior for problem2." << std::endl;
    problem2.SetPrior(prior_J, prior_r);

    // 第九步：设置求解器的相关参数，开始边缘化次新帧
    std::cout << "\nStep 9: Start marginalize subnew frame." << std::endl;
    problem2.Marginalize(problem2.GetCamerasNum() - 2, priorSize - 6);
    problem2.GetPrior(prior_J, prior_r);
    std::cout << "  prior J is\n" << prior_J << std::endl << "  prior r is\n" << prior_r << std::endl;

    return 0;
}