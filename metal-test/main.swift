import Foundation
import Metal

struct Dimension {
    var col: UInt32 = 0
    var row: UInt32 = 0
}

let device = MTLCreateSystemDefaultDevice()!
let library = device.newDefaultLibrary()!
let function = library.newFunctionWithName("matmul2")!
let state = try! device.newComputePipelineStateWithFunction(function)

var ldim = Dimension(col:4, row:2)
var rdim = Dimension(col:2, row:4)

let queue = device.newCommandQueue()
let buffer = queue.commandBuffer()
let encoder = buffer.computeCommandEncoder()
encoder.setComputePipelineState(state)

var lmat = [Float](count:Int(ldim.row * ldim.col), repeatedValue:0)
var rmat = [Float](count:Int(rdim.row * rdim.col), repeatedValue:0)
var omat = [Float](count:Int(ldim.row * rdim.col), repeatedValue:0)
for (index, _) in lmat.enumerate() {
    lmat[index] = Float(index)
}
for (index, _) in rmat.enumerate() {
    rmat[index] = Float(index)
}

let lbuffer = device.newBufferWithBytes(&lmat, length: lmat.count*sizeof(Float), options: [])
encoder.setBuffer(lbuffer, offset: 0, atIndex: 0)
let rbuffer = device.newBufferWithBytes(&rmat, length: rmat.count*sizeof(Float), options: [])
encoder.setBuffer(rbuffer, offset: 0, atIndex: 1)

let ldimb = device.newBufferWithBytes(&ldim, length: sizeof(Dimension), options:[])
encoder.setBuffer(ldimb, offset: 0, atIndex: 2)
let rdimb = device.newBufferWithBytes(&rdim, length: sizeof(Dimension), options:[])
encoder.setBuffer(rdimb, offset: 0, atIndex: 3)

let obuffer = device.newBufferWithBytes(&omat, length: omat.count*sizeof(Float), options: [])
encoder.setBuffer(obuffer, offset: 0, atIndex: 4)

let threadsPerGroup = MTLSize(width:32,height:32,depth:1)
encoder.setThreadgroupMemoryLength(threadsPerGroup.width * threadsPerGroup.height * sizeof(Float), atIndex: 0)
encoder.setThreadgroupMemoryLength(threadsPerGroup.width * threadsPerGroup.height * sizeof(Float), atIndex: 1)

let numThreadgroups = MTLSize(width:(Int(rdim.col)+threadsPerGroup.width-1)/threadsPerGroup.width,
                              height:(Int(ldim.row)+threadsPerGroup.height-1)/threadsPerGroup.height,
                              depth:1)
encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

encoder.endEncoding()
let start = NSDate()
buffer.commit()
buffer.waitUntilCompleted()
let elapsed = NSDate().timeIntervalSinceDate(start)
print(elapsed)

let data = NSData(bytesNoCopy: obuffer.contents(), length: omat.count*sizeof(Float), freeWhenDone: false)
data.getBytes(&omat, length:omat.count*sizeof(Float))
print(omat)

