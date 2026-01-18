#pragma once
#include <vector>
#include <cstdint>
#include <Accelerate/Accelerate.h>

class FeatureExtractor {
public:
    FeatureExtractor();
    ~FeatureExtractor();

    /**
     * @brief 处理音频数据，生成 Log-Mel Spectrogram
     * * @param pcm_audio 原始音频数据 (必须是 16kHz, 单声道, 已归一化到 -1~1)
     * @return std::vector<float> 扁平化的 Mel 谱图 (大小: 1 * 80 * 3000)
     */
    std::vector<float> process(const std::vector<float> &pcm_audio);
private:
    // --- 核心常量 (Whisper Tiny/Base 配置) ---
    static constexpr int N_FFT = 400;
    static constexpr int HOP_LENGTH = 160;
    static constexpr int N_MELS = 80;
    static constexpr int SAMPLE_RATE = 16000;

    // --- 预计算数据 (Cache) ---
    std::vector<float> hann_window_; // 汉宁窗 (大小: 400)
    std::vector<float> mel_filters_; // Mel 滤波器组矩阵 (大小: 80 * 201)

    // --- vDSP 句柄 ---
    // 专门用于加速 DFT/FFT 的设置结构体
    vDSP_DFT_Setup dft_setup_ = nullptr;

    // --- 初始化辅助函数 ---
    void init_hann_window();
    void init_mel_filters();
};
