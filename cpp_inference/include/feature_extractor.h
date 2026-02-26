#pragma once
#include <vector>
#include <cstdint>
#include <Accelerate/Accelerate.h>

/**
 * FeatureExtractor — Kaldi-compatible Fbank extraction for SenseVoice
 *
 * Pipeline:  16 kHz PCM → framing (25 ms Hamming, 10 ms hop)
 *            → DFT (N=512) → power spectrum → mel filterbank (80 bins)
 *            → log
 *
 * LFR and CMVN are applied downstream in SenseVoiceEngine.
 */
class FeatureExtractor {
public:
    FeatureExtractor();
    ~FeatureExtractor();

    /**
     * @brief Extract 80-dim log-Fbank features (variable length)
     * @param pcm_audio  16 kHz mono float audio in [-1, 1]
     * @param out_n_frames  [out] number of frames produced
     * @return Flat vector of size (out_n_frames * 80), row-major [T, 80]
     */
    std::vector<float> process(const std::vector<float>& pcm_audio,
                               int& out_n_frames);

private:
    // --- Core constants (Kaldi fbank compatible) ---
    static constexpr int N_FFT = 512;          // next power-of-2 ≥ 400
    static constexpr int FRAME_LENGTH = 400;   // 25 ms at 16 kHz
    static constexpr int HOP_LENGTH = 160;     // 10 ms at 16 kHz
    static constexpr int N_MELS = 80;
    static constexpr int SAMPLE_RATE = 16000;

    // --- Precomputed data ---
    std::vector<float> hamming_window_;  // size: FRAME_LENGTH (400)
    std::vector<float> mel_filters_;     // size: N_MELS * (N_FFT/2+1) = 80 * 257

    // --- DFT matrices (257 x 512) for matrix-based DFT ---
    std::vector<float> dft_mat_real_;
    std::vector<float> dft_mat_imag_;

    // --- Init helpers ---
    void init_hamming_window();
    void init_mel_filters();
    void init_dft_matrices();
};
