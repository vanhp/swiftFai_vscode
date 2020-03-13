/*
THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
file to edit: 02_fully_connected.ipynb

*/



import Path
import TensorFlow

public typealias TF=Tensor<Float>

public func normalize(_ x:TF, mean:TF, std:TF) -> TF {
    return (x-mean)/std
}

public func testNearZero(_ a: TF, tolerance: Float = 1e-3) {
    assert((abs(a) .< tolerance).all(), "Near zero: \(a)")
}

public func testSame(_ a: TF, _ b: TF) {
    // Check shapes match so broadcasting doesn't hide shape errors.
    assert(a.shape == b.shape)
    testNearZero(a-b)
}

public func mse(_ out: TF, _ targ: TF) -> TF {
    return (out.squeezingShape(at: -1) - targ).squared().mean()
}
