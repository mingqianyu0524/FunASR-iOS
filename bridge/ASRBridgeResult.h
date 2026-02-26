#ifndef ASRBridgeResult_h
#define ASRBridgeResult_h

#import <Foundation/Foundation.h>

@interface ASRBridgeResult : NSObject

@property (nonatomic, strong) NSString *text;
@property (nonatomic) double fbankTimeMs;
@property (nonatomic) double lfrCmvnTimeMs;
@property (nonatomic) double onnxForwardTimeMs;
@property (nonatomic) double decodeTimeMs;
@property (nonatomic) double totalInferenceMs;
@property (nonatomic) double audioDurationMs;
@property (nonatomic) double rtf;
@property (nonatomic) int64_t memoryBytesUsed;

@end

#endif /* ASRBridgeResult_h */
