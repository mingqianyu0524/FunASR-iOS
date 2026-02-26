#ifndef ParaformerContext_h
#define ParaformerContext_h

#import <Foundation/Foundation.h>
#import "ASRBridgeResult.h"

@interface ParaformerContext : NSObject

- (instancetype)initWithModelPath:(NSString *)modelPath;

/// Offline transcription with metrics
- (ASRBridgeResult *)transcribeDataWithMetrics:(NSData *)pcmData;

/// Streaming interface
- (void)startStream;
- (NSString *)feedChunk:(NSData *)pcmData;
- (ASRBridgeResult *)endStreamWithMetrics;

@end

#endif /* ParaformerContext_h */
