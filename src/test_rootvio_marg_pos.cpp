#include <iostream>
#include <random>

#include <include/ba_solver/root_vio/problem_vio.hpp>

using Scalar = float;      // 方便切换浮点数精度

/* Frame : 保存每帧的姿态和观测 */
struct Frame {
    Frame(rootVIO::Matrix3<Scalar> R, rootVIO::Vector3<Scalar> t) : R_wb(R), q_wb(R), t_wb(t) {};
    rootVIO::Matrix3<Scalar> R_wb;
    rootVIO::Quaternion<Scalar> q_wb;
    rootVIO::Vector3<Scalar> t_wb;
    std::unordered_map<int, rootVIO::Vector3<Scalar>> featurePerId; // 该帧观测到的特征以及特征id
};

/* VB ： 保存每帧的速度和偏差 */
struct VelocityBias {
    VelocityBias(rootVIO::Vector3<Scalar> v_wb, rootVIO::Vector3<Scalar> ba, rootVIO::Vector3<Scalar> bg) :
        v_wb(v_wb), ba(ba), bg(bg) {};
    rootVIO::Vector3<Scalar> v_wb;
    rootVIO::Vector3<Scalar> ba;
    rootVIO::Vector3<Scalar> bg;
};


/* 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测*/
void GenerateSimData(std::vector<Frame> &cameraPoses,
                     std::vector<rootVIO::Vector3<Scalar>> &points,
                     std::vector<VelocityBias> &vbs) {
    int featureNums = 4;   // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 4;      // 相机数目

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
            rootVIO::Vector3<Scalar> Pc = cameraPoses[i].R_wb.transpose() * (Pw - cameraPoses[i].t_wb);
            if (Pc.z() < 1e-10) {
                continue;
            }
            Pc = Pc / Pc.z();  // 归一化图像平面
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            cameraPoses[i].featurePerId.insert(std::make_pair(j, Pc));
        }
    }

    for (size_t i = 0; i < poseNums; ++i) {
        rootVIO::Vector3<Scalar> v = rootVIO::Vector3<Scalar>::Ones() * 0.1;
        rootVIO::Vector3<Scalar> bias_a = v;
        rootVIO::Vector3<Scalar> bias_g = v;
        vbs.emplace_back(VelocityBias(v, bias_a, bias_g));
    }
}

int main() {
    std::cout << "Test rootVIO solver on Mono BA problem." << std::endl;

    // 第一步：准备测试数据
    std::cout << "\nStep 1: Prepare dataset." << std::endl;
    std::vector<Frame> cameras;
    std::vector<rootVIO::Vector3<Scalar>> points;
    std::vector<VelocityBias> vbs; 
    GenerateSimData(cameras, points, vbs);

    // 第二步：构造 BA 问题求解器，并添加相机与 IMU 外参节点
    std::cout << "\nStep 2: Construct problem solver." << std::endl;
    rootVIO::ProblemVIO<Scalar> problem;
    problem.Reset();
    rootVIO::Quaternion<Scalar> q_bc;
    q_bc.setIdentity();
    rootVIO::Vector3<Scalar> t_bc = rootVIO::Vector3<Scalar>::Zero();
    std::shared_ptr<rootVIO::VertexExPose<Scalar>> exPose(new rootVIO::VertexExPose<Scalar>(q_bc, t_bc));
    problem.AddExPose(exPose);
    std::cout << "problem has 1 ex pose." << std::endl;

    // 第三步：构造相机 pose 节点，并添加到 problem 中
    std::cout << "\nStep 3: Add camera pose vertices." << std::endl;
    std::default_random_engine generator;
    std::normal_distribution<double> camera_rotation_noise(0., 0.1);
    std::normal_distribution<double> camera_position_noise(0., 0.2);
    std::vector<std::shared_ptr<rootVIO::VertexCameraPose<Scalar>>> cameraVertices;
    for (size_t i = 0; i < cameras.size(); ++i) {
        // 给相机位置和姿态初值增加噪声
        rootVIO::Matrix3<Scalar> noise_R, noise_X, noise_Y, noise_Z;
        noise_X = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitX());
        noise_Y = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitY());
        noise_Z = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), rootVIO::Vector3<Scalar>::UnitZ());
        noise_R = noise_X * noise_Y * noise_Z;
        rootVIO::Vector3<Scalar> noise_t(camera_position_noise(generator), camera_position_noise(generator), camera_position_noise(generator));
        rootVIO::Quaternion<Scalar> noise_q_wc(noise_R);
        if (i < 2) {
            noise_t.setZero();
            noise_q_wc.setIdentity();
        }

        std::shared_ptr<rootVIO::VertexCameraPose<Scalar>> cameraVertex(
            new rootVIO::VertexCameraPose<Scalar>(cameras[i].q_wb * noise_q_wc, cameras[i].t_wb + noise_t));
        cameraVertices.emplace_back(cameraVertex);
        problem.AddCamera(cameraVertex);
    }
    std::cout << "problem has " << problem.GetCamerasNum() << " cameras." << std::endl;

    // 第四步：构造特征点 position 节点，并添加到 problem 中
    std::cout << "\nStep 4: Add landmark position vertices." << std::endl;
    std::list<std::shared_ptr<rootVIO::VertexLandmark<Scalar>>> landmarkVertices;
    size_t maxObserveNum = cameras.size() / 2;
    for (size_t i = 0; i < points.size(); ++i) {
        // 为初值添加随机噪声
        std::normal_distribution<double> landmark_position_noise(0., 0.5);
        rootVIO::Vector3<Scalar> noise(landmark_position_noise(generator), landmark_position_noise(generator), landmark_position_noise(generator));
        std::shared_ptr<rootVIO::VertexLandmark<Scalar>> landmarkVertex(new rootVIO::VertexLandmark<Scalar>(points[i] + noise));
        // 构造观测
        std::unordered_map<size_t, std::shared_ptr<rootVIO::CameraObserve<Scalar>>> observes;
        ++maxObserveNum;
        if (maxObserveNum == cameras.size() + 1) {
            maxObserveNum = cameras.size() / 2;
        }
        for (size_t j = 0; j < cameras.size(); ++j) {
            std::shared_ptr<rootVIO::HuberKernel<Scalar>> kernel(new rootVIO::HuberKernel<Scalar>(1.0));
            std::shared_ptr<rootVIO::CameraObserve<Scalar>> observe(new rootVIO::CameraObserve<Scalar>(
                cameraVertices[j], cameras[j].featurePerId.find(i)->second.head<2>(), kernel));
            observes.insert(std::make_pair(j, observe));
        }
        
        landmarkVertices.emplace_back(landmarkVertex);
        problem.AddLandmark(landmarkVertex, observes, 3);
    }
    std::cout << "problem has " << problem.GetLandmarksNum() << " landmarks." << std::endl;

    // 第五步：构造 velocity bias 节点，添加到 problem 中
    std::cout << "\nStep 5: Add velocity bias vertices." << std::endl;
    std::vector<std::shared_ptr<rootVIO::VertexVelocityBias<Scalar>>> vbVertices;
    for (size_t i = 0; i < vbs.size(); ++i) {
        std::shared_ptr<rootVIO::VertexVelocityBias<Scalar>> vbVertex(new rootVIO::VertexVelocityBias<Scalar>(
            vbs[i].v_wb, vbs[i].ba, vbs[i].bg));
        vbVertices.emplace_back(vbVertex);
        problem.AddVelocityBias(vbVertex);
    }
    std::cout << "problem has " << problem.GetVelocityBiasesNum() << " velocity bias vertices." << std::endl;

    // 第六步：添加 IMU 残差约束
    std::cout << "\nStep 6: Add IMU factors." << std::endl;
    for (size_t i = 1; i < cameras.size(); ++i) {
        std::vector<std::shared_ptr<rootVIO::VertexCameraPose<Scalar>>> cameras;
        std::vector<std::shared_ptr<rootVIO::VertexVelocityBias<Scalar>>> velocityBiases;
        rootVIO::IMUBlock<Scalar>::LinearlizedPoint linear;
        rootVIO::IMUBlock<Scalar>::IMUJacobians jacobians;
        rootVIO::Vector3<Scalar> g_w;
        Scalar sumTime;
        rootVIO::IMUBlock<Scalar>::Order order;
        rootVIO::Matrix15<Scalar> cov;

        cameras.emplace_back(cameraVertices[i - 1]);
        cameras.emplace_back(cameraVertices[i]);
        velocityBiases.emplace_back(vbVertices[i - 1]);
        velocityBiases.emplace_back(vbVertices[i]);
        linear.bias_a = vbs[i].ba;
        linear.bias_g = vbs[i].bg;
        linear.delta_p.setZero();
        linear.delta_r.setIdentity();
        linear.delta_v.setZero();
        jacobians.dp_dba.setIdentity();
        jacobians.dp_dbg.setIdentity();
        jacobians.dr_dbg.setIdentity();
        jacobians.dv_dba.setIdentity();
        jacobians.dv_dbg.setIdentity();
        g_w << 0, 0, 9.8;
        sumTime = 0.2;
        order = {.P = 0, .R = 3, .V = 6, .BA = 9, .BG = 12};
        cov.setIdentity();
        cov = cov * 1e-4;

        std::shared_ptr<rootVIO::IMUBlock<Scalar>> imu(new rootVIO::IMUBlock<Scalar>(
            cameras, velocityBiases, i, linear, jacobians, g_w, sumTime, order, cov
        ));
        problem.AddIMUFactor(imu);
    }
    std::cout << "problem has " << problem.GetIMUBlocksNum() << " imu blocks." << std::endl;

    // 第七步：设置求解器的相关参数，开始边缘化
    std::cout << "\nStep 7: Start solve problem." << std::endl;
    size_t priorSize = cameras.size() * 6 + vbs.size() * 9 + 6 - 15;
    problem.Marginalize(0, priorSize);
    rootVIO::MatrixX<Scalar> prior_J;
    rootVIO::VectorX<Scalar> prior_r;
    problem.GetPrior(prior_J, prior_r);
    std::cout << "  prior J is\n" << prior_J << std::endl << "  prior r is\n" << prior_r << std::endl;

    return 0;
}