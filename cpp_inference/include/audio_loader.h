#include <vector>
#include <string>

class AudioLoader {
public:
    // 目标采样率
    static constexpr int TARGET_SAMPLE_RATE = 16000;

    // 读取 WAV 文件，并自动转换为:
    // 1. 浮点数 (float)
    // 2. 单声道 (Mono)
    // 3. 16kHz
    // 4. 归一化 (-1.0 ~ 1.0)
    static std::vector<float> load(const std::string& filename);
};