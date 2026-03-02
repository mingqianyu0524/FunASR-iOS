import XCTest

class ASRPipelineTests: XCTestCase {

    /// Model directory: CI injects via MODEL_DIR env var; local falls back to app bundle.
    var modelDir: String {
        ProcessInfo.processInfo.environment["MODEL_DIR"]
            ?? Bundle.main.resourcePath!
    }

    func testSenseVoiceSample() throws {
        // Load WAV fixture from test bundle
        let pcmData = try loadFixtureWAV("ramc_sample")
        let ref = try loadFixtureText("ramc_sample_ref")

        // Initialize engine
        guard let ctx = SenseVoiceContext(modelPath: modelDir) else {
            XCTFail("SenseVoice init failed, MODEL_DIR=\(modelDir)")
            return
        }

        // Run inference
        guard let result = ctx.transcribeData(withMetrics: pcmData) else {
            XCTFail("transcribeDataWithMetrics returned nil")
            return
        }

        let text = stripSpecialTokens(result.text ?? "")
        let cer = CERHelper.cer(hypothesis: text, reference: ref)

        print("""
        ── SenseVoice Pipeline Test ──
        Hypothesis: \(text)
        Reference:  \(ref)
        CER:        \(String(format: "%.1f%%", cer * 100))
        RTF:        \(String(format: "%.3f", result.rtf))
        Inference:  \(String(format: "%.0f ms", result.totalInferenceMs))
        ──────────────────────────────
        """)

        XCTAssertFalse(text.isEmpty, "Transcription should not be empty")
        XCTAssertLessThan(cer, 0.20, "CER \(cer) exceeds 20% threshold")
    }

    // MARK: - Helpers

    /// Load a WAV file from the test bundle, decode 16-bit PCM to [Float], return as Data.
    private func loadFixtureWAV(_ name: String) throws -> Data {
        let bundle = Bundle(for: type(of: self))
        let url = try XCTUnwrap(
            bundle.url(forResource: name, withExtension: "wav"),
            "Fixture \(name).wav not found in test bundle"
        )

        let fileData = try Data(contentsOf: url)

        // Parse WAV header to find data chunk
        // Standard WAV: 44-byte header, but we parse properly for robustness
        let headerSize = parseWAVDataOffset(fileData) ?? 44
        let pcmBytes = fileData.dropFirst(headerSize)

        return pcmBytes.withUnsafeBytes { rawBuffer in
            let int16Samples = rawBuffer.bindMemory(to: Int16.self)
            var floats = [Float](repeating: 0, count: int16Samples.count)
            for i in floats.indices {
                floats[i] = Float(int16Samples[i]) / 32768.0
            }
            return floats.withUnsafeBufferPointer { Data(buffer: $0) }
        }
    }

    /// Find the byte offset of the "data" chunk payload in a WAV file.
    private func parseWAVDataOffset(_ data: Data) -> Int? {
        guard data.count > 44 else { return nil }
        // Search for "data" marker
        let marker: [UInt8] = [0x64, 0x61, 0x74, 0x61] // "data"
        for i in 0..<(data.count - 8) {
            if data[i] == marker[0] && data[i+1] == marker[1]
                && data[i+2] == marker[2] && data[i+3] == marker[3] {
                return i + 8 // skip "data" + 4-byte chunk size
            }
        }
        return nil
    }

    /// Strip SenseVoice special tokens like <|zh|>, <|NEUTRAL|>, <|Speech|> from decoded text.
    private func stripSpecialTokens(_ text: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: "<\\|[^|>]*\\|>") else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, range: range, withTemplate: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func loadFixtureText(_ name: String) throws -> String {
        let bundle = Bundle(for: type(of: self))
        let url = try XCTUnwrap(
            bundle.url(forResource: name, withExtension: "txt"),
            "Fixture \(name).txt not found in test bundle"
        )
        return try String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
