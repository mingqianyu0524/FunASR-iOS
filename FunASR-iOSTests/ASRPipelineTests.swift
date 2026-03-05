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

    // MARK: - Batch Evaluation

    /// Batch evaluation across RAMC and AISHELL-1 CI fixtures.
    /// Requires EVAL_AUDIO_DIR env var pointing to ci_fixtures/ directory.
    /// Skipped automatically if the env var is not set.
    func testBatchEval() throws {
        // Env var set via Xcode scheme (local dev).
        // Fallback: file written by CI before xcodebuild — iOS Simulator maps host /tmp directly.
        let evalAudioDir: String
        if let v = ProcessInfo.processInfo.environment["EVAL_AUDIO_DIR"], !v.isEmpty {
            evalAudioDir = v
        } else if let v = try? String(contentsOfFile: "/tmp/funasr_eval_dir.txt", encoding: .utf8),
                  !v.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            evalAudioDir = v.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            throw XCTSkip("EVAL_AUDIO_DIR not set — skipping batch evaluation")
        }

        guard let ctx = SenseVoiceContext(modelPath: modelDir) else {
            XCTFail("SenseVoice init failed, MODEL_DIR=\(modelDir)")
            return
        }

        let subsets: [(name: String, subdir: String)] = [
            ("RAMC", "ramc"),
            ("AISHELL-1", "aishell1"),
        ]

        // Collect all subset results for JSON output
        var allResults: [[String: Any]] = []

        for subset in subsets {
            let subsetDir = URL(fileURLWithPath: evalAudioDir).appendingPathComponent(subset.subdir)
            let wavFiles = (try? FileManager.default.contentsOfDirectory(
                at: subsetDir, includingPropertiesForKeys: nil
            ).filter { $0.pathExtension == "wav" }.sorted { $0.lastPathComponent < $1.lastPathComponent }) ?? []

            guard !wavFiles.isEmpty else {
                print("── Batch Eval: \(subset.name): no WAV files found in \(subsetDir.path), skipping")
                continue
            }

            var totalRefChars = 0
            var totalDist = 0
            var totalDel = 0, totalIns = 0, totalSub = 0
            var errorUtts = 0
            var totalUtts = 0

            for wavURL in wavFiles {
                let stem = wavURL.deletingPathExtension().lastPathComponent
                let txtURL = subsetDir.appendingPathComponent(stem + ".txt")
                guard let ref = try? String(contentsOf: txtURL, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                      !ref.isEmpty else { continue }

                guard let pcmData = try? loadWAVFromURL(wavURL) else { continue }
                guard let result = ctx.transcribeData(withMetrics: pcmData) else { continue }

                let hyp = stripSpecialTokens(result.text ?? "")
                let details = CERHelper.cerWithDetails(hypothesis: hyp, reference: ref)
                let refChars = ref.filter { !$0.isWhitespace }.count

                totalRefChars += refChars
                let dist = details.deletions + details.insertions + details.substitutions
                totalDist += dist
                totalDel += details.deletions
                totalIns += details.insertions
                totalSub += details.substitutions
                if dist > 0 { errorUtts += 1 }
                totalUtts += 1
            }

            guard totalUtts > 0, totalRefChars > 0 else {
                print("── Batch Eval: \(subset.name): no matched utterances, skipping")
                continue
            }

            let cer = Double(totalDist) / Double(totalRefChars)
            let ser = Double(errorUtts) / Double(totalUtts)

            print("""
            ── Batch Eval: \(subset.name) (\(totalUtts) utterances) ──
            CER:   \(String(format: "%.1f%%", cer * 100))   SER:  \(String(format: "%.1f%%", ser * 100))
            D:  \(totalDel)  I:  \(totalIns)  S: \(totalSub)  Ref chars: \(totalRefChars)
            """)

            allResults.append([
                "dataset": subset.name,
                "utterances": totalUtts,
                "cer": cer,
                "ser": ser,
                "deletions": totalDel,
                "insertions": totalIns,
                "substitutions": totalSub,
                "ref_chars": totalRefChars,
            ])

            // Loose regression guard: CER < 30% for both datasets
            XCTAssertLessThan(cer, 0.30, "\(subset.name) CER \(String(format:"%.1f%%",cer*100)) exceeds 30% threshold")
        }

        // Write results JSON to EVAL_AUDIO_DIR for parse_test_results.py
        if !allResults.isEmpty,
           let data = try? JSONSerialization.data(withJSONObject: allResults, options: .prettyPrinted) {
            let outURL = URL(fileURLWithPath: evalAudioDir).appendingPathComponent("batch_eval_results.json")
            try? data.write(to: outURL)
        }
    }

    // MARK: - Paraformer Batch Evaluation

    /// Batch evaluation of Paraformer across RAMC and AISHELL-1 CI fixtures.
    /// Requires EVAL_AUDIO_DIR env var (or /tmp/funasr_eval_dir.txt fallback).
    /// Skipped automatically if neither is set.
    func testParaformerBatchEval() throws {
        let evalAudioDir: String
        if let v = ProcessInfo.processInfo.environment["EVAL_AUDIO_DIR"], !v.isEmpty {
            evalAudioDir = v
        } else if let v = try? String(contentsOfFile: "/tmp/funasr_eval_dir.txt", encoding: .utf8),
                  !v.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            evalAudioDir = v.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            throw XCTSkip("EVAL_AUDIO_DIR not set — skipping Paraformer batch evaluation")
        }

        guard let ctx = ParaformerContext(modelPath: modelDir) else {
            XCTFail("Paraformer init failed, MODEL_DIR=\(modelDir)")
            return
        }

        let subsets: [(name: String, subdir: String)] = [
            ("RAMC [Paraformer]", "ramc"),
            ("AISHELL-1 [Paraformer]", "aishell1"),
        ]

        var allResults: [[String: Any]] = []

        for subset in subsets {
            let subsetDir = URL(fileURLWithPath: evalAudioDir).appendingPathComponent(subset.subdir)
            let wavFiles = (try? FileManager.default.contentsOfDirectory(
                at: subsetDir, includingPropertiesForKeys: nil
            ).filter { $0.pathExtension == "wav" }.sorted { $0.lastPathComponent < $1.lastPathComponent }) ?? []

            guard !wavFiles.isEmpty else {
                print("── Batch Eval: \(subset.name): no WAV files found in \(subsetDir.path), skipping")
                continue
            }

            var totalRefChars = 0
            var totalDist = 0
            var totalDel = 0, totalIns = 0, totalSub = 0
            var errorUtts = 0
            var totalUtts = 0

            for wavURL in wavFiles {
                let stem = wavURL.deletingPathExtension().lastPathComponent
                let txtURL = subsetDir.appendingPathComponent(stem + ".txt")
                guard let ref = try? String(contentsOf: txtURL, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                      !ref.isEmpty else { continue }

                guard let pcmData = try? loadWAVFromURL(wavURL) else { continue }
                guard let result = ctx.transcribeData(withMetrics: pcmData) else { continue }

                // Paraformer does not use special tokens — no stripping needed
                let hyp = result.text ?? ""
                let details = CERHelper.cerWithDetails(hypothesis: hyp, reference: ref)
                let refChars = ref.filter { !$0.isWhitespace }.count

                totalRefChars += refChars
                let dist = details.deletions + details.insertions + details.substitutions
                totalDist += dist
                totalDel += details.deletions
                totalIns += details.insertions
                totalSub += details.substitutions
                if dist > 0 { errorUtts += 1 }
                totalUtts += 1
            }

            guard totalUtts > 0, totalRefChars > 0 else {
                print("── Batch Eval: \(subset.name): no matched utterances, skipping")
                continue
            }

            let cer = Double(totalDist) / Double(totalRefChars)
            let ser = Double(errorUtts) / Double(totalUtts)

            print("""
            ── Batch Eval: \(subset.name) (\(totalUtts) utterances) ──
            CER:   \(String(format: "%.1f%%", cer * 100))   SER:  \(String(format: "%.1f%%", ser * 100))
            D:  \(totalDel)  I:  \(totalIns)  S: \(totalSub)  Ref chars: \(totalRefChars)
            """)

            allResults.append([
                "dataset": subset.name,
                "utterances": totalUtts,
                "cer": cer,
                "ser": ser,
                "deletions": totalDel,
                "insertions": totalIns,
                "substitutions": totalSub,
                "ref_chars": totalRefChars,
            ])

            XCTAssertLessThan(cer, 0.30, "\(subset.name) CER \(String(format:"%.1f%%",cer*100)) exceeds 30% threshold")
        }

        // Append Paraformer results to the same batch_eval_results.json
        if !allResults.isEmpty {
            let outURL = URL(fileURLWithPath: evalAudioDir).appendingPathComponent("batch_eval_results.json")
            var combined: [[String: Any]] = []
            if let existing = try? Data(contentsOf: outURL),
               let decoded = try? JSONSerialization.jsonObject(with: existing) as? [[String: Any]] {
                combined = decoded
            }
            combined.append(contentsOf: allResults)
            if let data = try? JSONSerialization.data(withJSONObject: combined, options: .prettyPrinted) {
                try? data.write(to: outURL)
            }
        }
    }

    // MARK: - SNR Robustness (optional)

    /// Add Gaussian white noise at a given SNR (dB) to a float PCM array.
    private func addNoise(to pcm: [Float], snrDB: Float) -> [Float] {
        let signalPower = pcm.reduce(0) { $0 + $1 * $1 } / Float(pcm.count)
        guard signalPower > 0 else { return pcm }
        let noisePower = signalPower / pow(10, snrDB / 10)
        let noiseStd = sqrt(noisePower)
        // Simple LCG-based noise for reproducibility (no Foundation dependency)
        var state: UInt64 = 12345678
        func nextNormal() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u1 = Float(state >> 33) / Float(UInt32.max)
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u2 = Float(state >> 33) / Float(UInt32.max)
            // Box-Muller
            let r = sqrt(-2 * log(max(u1, 1e-10)))
            let theta = 2 * Float.pi * u2
            return r * cos(theta)
        }
        return pcm.map { $0 + nextNormal() * noiseStd }
    }

    // MARK: - Helpers

    /// Load WAV from an arbitrary URL (for batch eval directory scanning).
    private func loadWAVFromURL(_ url: URL) throws -> Data {
        let fileData = try Data(contentsOf: url)
        let headerSize = parseWAVDataOffset(fileData) ?? 44
        let pcmBytes = fileData.dropFirst(headerSize)
        return pcmBytes.withUnsafeBytes { rawBuffer in
            let int16Samples = rawBuffer.bindMemory(to: Int16.self)
            var floats = [Float](repeating: 0, count: int16Samples.count)
            for i in floats.indices { floats[i] = Float(int16Samples[i]) / 32768.0 }
            return floats.withUnsafeBufferPointer { Data(buffer: $0) }
        }
    }

    // MARK: - Original Helpers

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
