import Foundation
import Metal

struct Dimension {
    var col: UInt32 = 0
    var row: UInt32 = 0
}

let device = MTLCreateSystemDefaultDevice()!
let library = device.newDefaultLibrary()!
let function1 = library.newFunctionWithName("matmul")!
let function2 = library.newFunctionWithName("matmul2")!
let state1 = try! device.newComputePipelineStateWithFunction(function1)
let state2 = try! device.newComputePipelineStateWithFunction(function2)
let queue = device.newCommandQueue()
var ldim = Dimension(col:3491, row:31)
var rdim = Dimension(col:31, row:3491)
var lmat = [Float](count:Int(ldim.row * ldim.col), repeatedValue:0)
var rmat = [Float](count:Int(rdim.row * rdim.col), repeatedValue:0)
for (index, _) in lmat.enumerate() {
    lmat[index] = Float(index)
}
for (index, _) in rmat.enumerate() {
    rmat[index] = Float(index)
}

func matmul(state: MTLComputePipelineState) -> [Float] {
    var omat = [Float](count:Int(ldim.row * rdim.col), repeatedValue:0)
    let buffer = queue.commandBuffer()
    let encoder = buffer.computeCommandEncoder()
    encoder.setComputePipelineState(state)
    
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
    print("elapsed:", elapsed)

    let data = NSData(bytesNoCopy: obuffer.contents(), length: omat.count*sizeof(Float), freeWhenDone: false)
    data.getBytes(&omat, length:omat.count*sizeof(Float))
    return omat
}

func matmulCPU() -> [Float] {
    var omat = [Float](count:Int(ldim.row * rdim.col), repeatedValue:0)

    let start = NSDate()
    for var i = UInt32(0); i < ldim.row; i++ {
        for var j = UInt32(0); j < rdim.col; j++ {
            var v = Float(0)
            for var k = UInt32(0); k < ldim.col; k++ {
                v += lmat[Int(i * ldim.col + k)] * rmat[Int(k * rdim.col + j)]
            }
            omat[Int(i * rdim.col + j)] = v
        }
    }
    let elapsed = NSDate().timeIntervalSinceDate(start)
    print("elapsed:", elapsed)
    
    return omat
}

let out1 = matmul(state1)
let out2 = matmul(state2)
let out3 = matmulCPU()

if out1.count != out2.count || out1.count != out3.count {
    print("something wrong")
    exit(0)
}

var pair1 = Zip2Sequence(out1, out2).generate()
while let elem = pair1.next() {
    let (l, r) = elem
    if l != r {
        print("something wrong")
        exit(0)
    }
}

var pair2 = Zip2Sequence(out1, out3).generate()
while let elem = pair2.next() {
    let (l, r) = elem
    if l != r {
        print("something wrong")
        exit(0)
    }
}


