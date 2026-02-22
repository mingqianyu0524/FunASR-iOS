//
//  SenseVoiceContext.mm
//  WhisperIOS
//
//  Obj-C++ bridge for SenseVoiceEngine
//

#import <Foundation/Foundation.h>
#include <vector>
#import "SenseVoiceContext.h"
#include "sensevoice_engine.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

@interface SenseVoiceContext () {
    SenseVoiceEngine *engine;
    BOOL initialized;
}
@end

@implementation SenseVoiceContext

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        initialized = NO;
        try {
            std::string base_path([modelPath UTF8String]);
            engine = new SenseVoiceEngine();

            std::string model_path = base_path + "/model.int8.onnx";
            std::string tokens_path = base_path + "/tokens.txt";

            bool success = engine->initialize(model_path, tokens_path);
            if (success) {
                initialized = YES;
                NSLog(@"✅ SenseVoiceContext: Engine initialized successfully.");
            } else {
                NSLog(@"❌ SenseVoiceContext: Engine initialization failed.");
            }
        } catch (const std::exception& e) {
            NSLog(@"❌ SenseVoiceContext init exception: %s", e.what());
        } catch (...) {
            NSLog(@"❌ SenseVoiceContext init: unknown exception");
        }
    }
    return self;
}

- (void)dealloc {
    delete engine;
}

- (NSString *)transcribeData:(NSData *)pcmData {
    if (!initialized || !engine) {
        NSLog(@"❌ SenseVoiceContext: Engine not initialized, cannot transcribe.");
        return @"Error: Engine not initialized.";
    }

    try {
        // Zero-copy: directly interpret NSData bytes as float array
        NSUInteger sampleCount = [pcmData length] / sizeof(float);
        const float* rawPtr = (const float *)[pcmData bytes];

        // Construct vector from raw pointer (single copy)
        std::vector<float> pcm_data(rawPtr, rawPtr + sampleCount);

        NSLog(@"🧠 SenseVoiceContext: Running transcribe with %lu samples...",
              (unsigned long)sampleCount);
        std::string result = engine->transcribe(pcm_data);
        return [NSString stringWithUTF8String:result.c_str()];
    } catch (const Ort::Exception& e) {
        NSLog(@"❌ SenseVoiceContext ONNX Runtime error: %s", e.what());
        return [NSString stringWithFormat:@"ONNX Error: %s", e.what()];
    } catch (const std::exception& e) {
        NSLog(@"❌ SenseVoiceContext exception: %s", e.what());
        return [NSString stringWithFormat:@"Error: %s", e.what()];
    } catch (...) {
        NSLog(@"❌ SenseVoiceContext: unknown exception during transcribe");
        return @"Error: Unknown exception during transcription.";
    }
}

@end
