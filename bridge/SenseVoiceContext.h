//
//  SenseVoiceContext.h
//  WhisperIOS
//
//  Obj-C bridge for SenseVoiceEngine (C++ → Swift)
//

#ifndef SenseVoiceContext_h
#define SenseVoiceContext_h

#import <Foundation/Foundation.h>
#import "ASRBridgeResult.h"

@interface SenseVoiceContext : NSObject

- (instancetype)initWithModelPath:(NSString *)modelPath;

/// Transcribe audio from raw float PCM data passed as NSData (zero-copy from Swift)
- (NSString *)transcribeData:(NSData *)pcmData;

/// Transcribe with metrics collection
- (ASRBridgeResult *)transcribeDataWithMetrics:(NSData *)pcmData;

@end

#endif /* SenseVoiceContext_h */
