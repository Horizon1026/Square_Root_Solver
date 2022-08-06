# cmake工程最低版本为3.10
cmake_minimum_required( VERSION 3.10 )

# cmake工程名
project( rootBA )

# 设置编译类型为发布
set( CMAKE_BUILD_TYPE "Release" )

# 添加c++ 17标准的支持
set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

# 寻找 TBB 库
find_package( TBB REQUIRED )

# 寻找 Eigen 库并添加其路径
include_directories( "/usr/include/eigen3" )

# 添加自定义路径
include_directories( ${PROJECT_SOURCE_DIR} )
include_directories( ${PROJECT_SOURCE_DIR}/include )
include_directories( ${PROJECT_SOURCE_DIR}/test )
include_directories( ${CERES_INCLUDE_DIRS} )

# 添加可执行文件 test_rootvio_pos
add_executable( test_rootvio_pos
    src/test_rootvio_pos.cpp
)
# 链接库到可执行文件
target_link_libraries( test_rootvio_pos
    TBB::tbb
)

# 添加可执行文件 test_rootvio_invdep
add_executable( test_rootvio_invdep
    src/test_rootvio_invdep.cpp
)
# 链接库到可执行文件
target_link_libraries( test_rootvio_invdep
    TBB::tbb
)

# 添加可执行文件 test_rootvio_marg_pos
add_executable( test_rootvio_marg_pos
    src/test_rootvio_marg_pos.cpp
)
# 链接库到可执行文件
target_link_libraries( test_rootvio_marg_pos
    TBB::tbb
)

# 添加可执行文件 test_rootvo_pos
add_executable( test_rootvo_pos
    src/test_rootvo_pos.cpp
)
# 链接库到可执行文件
target_link_libraries( test_rootvo_pos
    TBB::tbb
)

# 添加可执行文件 test_rootvo_invdep
add_executable( test_rootvo_invdep
    src/test_rootvo_invdep.cpp
)
# 链接库到可执行文件
target_link_libraries( test_rootvo_invdep
    TBB::tbb
)

# 添加可执行文件 test_rootvo_marg_pos
add_executable( test_rootvo_marg_pos
    src/test_rootvo_marg_pos.cpp
)
# 链接库到可执行文件
target_link_libraries( test_rootvo_marg_pos
    TBB::tbb
)

# 添加可执行文件 test_rootvo_marg_invdep
add_executable( test_rootvo_marg_invdep
    src/test_rootvo_marg_invdep.cpp
)
# 链接库到可执行文件
target_link_libraries( test_rootvo_marg_invdep
    TBB::tbb
)