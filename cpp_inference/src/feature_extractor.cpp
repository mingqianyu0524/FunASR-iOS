#include "feature_extractor.h"
#include <algorithm> // for std::min, std::copy, std::fill
#include <iostream>  // for debug prints
#include <cmath>     // for std::abs, cos, sin

// --- [Patch] Static Global DFT Matrices Fix ---
// 使用函数内部的静态变量 (Meyers' Singleton) 来解决全局初始化顺序问题。
// 这样可以确保 vector 不会被重复构造或清空。

std::vector<float>& get_dft_mat_real() {
    static std::vector<float> mat;
    return mat;
}

std::vector<float>& get_dft_mat_imag() {
    static std::vector<float> mat;
    return mat;
}

// Helper to initialize DFT matrices (201 x 400)
void init_dft_matrices(int n_fft) {
    auto& mat_real = get_dft_mat_real();
    auto& mat_imag = get_dft_mat_imag();

    if (!mat_real.empty()) return; // Already initialized

    int n_bins = n_fft / 2 + 1;
    mat_real.resize(n_bins * n_fft);
    mat_imag.resize(n_bins * n_fft);

    const float PI = 3.14159265358979323846f;

    for (int k = 0; k < n_bins; ++k) {
        for (int n = 0; n < n_fft; ++n) {
            float angle = 2.0f * PI * k * n / n_fft;
            // Row-major storage: index = k * n_fft + n
            // Real part: cos(angle)
            mat_real[k * n_fft + n] = std::cos(angle);
            // Imag part: -sin(angle)
            mat_imag[k * n_fft + n] = -std::sin(angle);
        }
    }
    std::cout << "✅ Matrix DFT initialized (Size: " << n_bins << "x" << n_fft << ")" << std::endl;
}
// ------------------------------------------------------------------

FeatureExtractor::FeatureExtractor()
{
    // 1. Init Hann Window
    init_hann_window();

    // 2. Init Mel Filterbank
    init_mel_filters();
    
    // 3. Init Matrix DFT (Replacement for vDSP DFT Setup)
    init_dft_matrices(N_FFT);
    
    // We no longer use dft_setup_ because N=400 is not supported by vDSP DFT
    dft_setup_ = nullptr; 
}

FeatureExtractor::~FeatureExtractor()
{
    // dft_setup_ is not used
}

void FeatureExtractor::init_hann_window()
{
    hann_window_.resize(N_FFT);
    vDSP_hann_window(hann_window_.data(), N_FFT, 0);
}

void FeatureExtractor::init_mel_filters()
{
    constexpr int32_t n_sample = 82;
    constexpr float m_min = 0.0f;
    float m_max = 2595.0f * std::log10(1.0f + 8000.0f/700.0f);
    
    std::vector<float> points(n_sample);
    vDSP_vgen(&m_min, &m_max, points.data(), 1, n_sample);
    
    std::for_each(points.begin(), points.end(), [](float& m) {
        m = 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f);
        m *= (static_cast<float>(N_FFT) / SAMPLE_RATE);
    });

    const int32_t n_spectrum_bins = (N_FFT / 2) + 1; // 201
    
    mel_filters_.resize(N_MELS * n_spectrum_bins);
    
    for (int i = 0; i < N_MELS; ++i) {
        float left   = points[i];
        float center = points[i+1];
        float right  = points[i+2];

        float norm_factor = 2.0f / (points[i+2] - points[i]);

        for (int j = 0; j < n_spectrum_bins; ++j) {
            float weight = 0.0f;
            float fj = static_cast<float>(j);

            if (fj > left && fj <= center) {
                weight = (fj - left) / (center - left);
            } else if (fj > center && fj < right) {
                weight = (right - fj) / (right - center);
            }
            
            weight *= norm_factor;
            mel_filters_[i * n_spectrum_bins + j] = weight;
        }
    }
}

std::vector<float> FeatureExtractor::process(const std::vector<float> &pcm_audio)
{
    std::cout << "[DEBUG] FeatureExtractor::process start." << std::endl;

    const int32_t n_frames = 3000;
    const int32_t n_spectrum_bins = (N_FFT / 2) + 1;
    
    std::vector<float> power_spectrum(n_frames * n_spectrum_bins, 0.0f);

    std::vector<float> frame(N_FFT);
    std::vector<float> windowed_frame(N_FFT);
    // output_real/imag size is n_spectrum_bins (201), not N_FFT
    std::vector<float> output_real(n_spectrum_bins); 
    std::vector<float> output_imag(n_spectrum_bins);
    std::vector<float> magnitudes(n_spectrum_bins);
    
    // Temp complex for zvmags
    DSPSplitComplex temp_complex;
    temp_complex.realp = output_real.data();
    temp_complex.imagp = output_imag.data();

    // 获取 DFT 矩阵的引用
    auto& dft_real = get_dft_mat_real();
    auto& dft_imag = get_dft_mat_imag();

    // 安全检查：防止矩阵未初始化
    if (dft_real.empty() || dft_imag.empty()) {
        std::cerr << "❌ CRITICAL: DFT Matrices empty! Re-initializing..." << std::endl;
        init_dft_matrices(N_FFT);
    }

    std::cout << "[DEBUG] Starting STFT loop (3000 iterations)..." << std::endl;

    for (int k = 0; k < n_frames; k++) {
        bool debug_step = (k == 0); 

        int sample_idx = k * HOP_LENGTH;

        // 1. Framing
        std::fill(frame.begin(), frame.end(), 0.0f);
        if (sample_idx < (int)pcm_audio.size()) {
            size_t copy_count = std::min((size_t)N_FFT, pcm_audio.size() - sample_idx);
            std::copy(pcm_audio.begin() + sample_idx, 
                      pcm_audio.begin() + sample_idx + copy_count, 
                      frame.begin());
        }

        // 2. Windowing
        if (debug_step) std::cout << "  [Step] Windowing..." << std::endl;
        vDSP_vmul(frame.data(), 1, hann_window_.data(), 1, windowed_frame.data(), 1, N_FFT);

        // 3. DFT (Matrix Multiplication)
        if (debug_step) std::cout << "  [Step] Matrix DFT..." << std::endl;
        
        // Calculate Real Part: output_real (201x1) = dft_real (201x400) * windowed_frame (400x1)
        vDSP_mmul(dft_real.data(), 1, windowed_frame.data(), 1, output_real.data(), 1, 
                  n_spectrum_bins, 1, N_FFT);
                  
        // Calculate Imag Part: output_imag (201x1) = dft_imag (201x400) * windowed_frame (400x1)
        vDSP_mmul(dft_imag.data(), 1, windowed_frame.data(), 1, output_imag.data(), 1, 
                  n_spectrum_bins, 1, N_FFT);

        // 4. Power
        if (debug_step) std::cout << "  [Step] vDSP_zvmags..." << std::endl;
        vDSP_zvmags(&temp_complex, 1, magnitudes.data(), 1, n_spectrum_bins);
        
        // 5. Store
        auto dest_iter = power_spectrum.begin() + (k * n_spectrum_bins);
        std::copy(magnitudes.begin(), magnitudes.end(), dest_iter);
    }
    std::cout << "[DEBUG] STFT Loop finished." << std::endl;

    // Transpose
    std::cout << "[DEBUG] Transposing..." << std::endl;
    std::vector<float> power_spectrum_T(n_spectrum_bins * n_frames);
    vDSP_mtrans(power_spectrum.data(), 1, power_spectrum_T.data(), 1, n_spectrum_bins, n_frames);

    // Matrix Mul
    std::cout << "[DEBUG] Matrix Multiplication..." << std::endl;
    std::vector<float> log_mel_spec(N_MELS * n_frames);
    
    if (mel_filters_.size() != (size_t)(N_MELS * n_spectrum_bins)) {
        std::cerr << "❌ MEL FILTER SIZE WRONG: " << mel_filters_.size() << std::endl;
        exit(1);
    }

    vDSP_mmul(
        mel_filters_.data(), 1,      
        power_spectrum_T.data(), 1,  
        log_mel_spec.data(), 1,      
        N_MELS,                      
        n_frames,                    
        n_spectrum_bins              
    );

    // Post processing
    std::cout << "[DEBUG] Post-processing (Log & Scale)..." << std::endl;
    float epsilon = 1e-10f;
    int n_total = static_cast<int>(log_mel_spec.size());
    
    vDSP_vthr(log_mel_spec.data(), 1, &epsilon, log_mel_spec.data(), 1, n_total);
    vvlog10f(log_mel_spec.data(), log_mel_spec.data(), &n_total);

    float max_val = 0.0f;
    vDSP_maxv(log_mel_spec.data(), 1, &max_val, log_mel_spec.size());
    float min_val = max_val - 8.0f;
    vDSP_vthr(log_mel_spec.data(), 1, &min_val, log_mel_spec.data(), 1, log_mel_spec.size());
    float n_factor = 4.0f;
    vDSP_vsadd(log_mel_spec.data(), 1, &n_factor, log_mel_spec.data(), 1, log_mel_spec.size());
    vDSP_vsdiv(log_mel_spec.data(), 1, &n_factor, log_mel_spec.data(), 1, log_mel_spec.size());

    // --- 🔍 添加这段诊断代码 ---
    std::cout << "🔍 Mel Spectrogram Stats:" << std::endl;
    std::cout << "   Global Max Value (before scaling): " << max_val << std::endl;
    std::cout << "   First 10 values: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << log_mel_spec[i] << ", ";
    }
    std::cout << std::endl;
    // -------------------------

    std::cout << "[DEBUG] Feature extraction done." << std::endl;
    return log_mel_spec;
}