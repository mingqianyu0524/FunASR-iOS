cmake_minimum_required(VERSION 3.10)
project(WhisperNative)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找 onnxruntime
# Homebrew 安装的通常在 /usr/local 或 /opt/homebrew
# 我们直接让 pkg-config 或 find_library 找，或者手动指定路径
# 这里用最通用的 brew 路径猜测法

if(EXISTS "/opt/homebrew/include/onnxruntime")
    include_directories("/opt/homebrew/include/onnxruntime")
    link_directories("/opt/homebrew/lib")
elseif(EXISTS "/usr/local/include/onnxruntime")
    include_directories("/usr/local/include/onnxruntime")
    link_directories("/usr/local/lib")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# 查找 CoreFoundation (Mac系统库，ORT依赖)
find_library(CORE_FOUNDATION_LIB CoreFoundation)
find_library(ACCELERATE_LIB Accelerate)

add_executable(whisper_inference
    src/audio_loader.cpp
    src/main_inference.cpp 
    src/whisper_tokenizer.cpp
    src/feature_extractor.cpp
    src/whisper_engine.cpp 
)

# 链接 onnxruntime
target_link_libraries(whisper_inference onnxruntime ${CORE_FOUNDATION_LIB} ${ACCELERATE_LIB})