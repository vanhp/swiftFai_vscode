import Foundation
import TensorFlow
import fastai_layers

struct why_sqrt5 {
    var text = "Hello, why_sqrt5!"
}

// ## Does nn.Conv2d init work well?
public extension Tensor where Scalar: TensorFlowFloatingPoint {
    func stats() -> (mean: Tensor, std: Tensor) {
        return (mean: mean(), std: standardDeviation())
    }
}
// This is in 1a now so this code is disabled from here:

// ```swift
// var rng = PhiloxRandomNumberGenerator.global

// extension Tensor where Scalar: TensorFlowFloatingPoint {
//     init(kaimingNormal shape: TensorShape, negativeSlope: Double = 1.0) {
//         // Assumes Leaky ReLU nonlinearity
//         let gain = Scalar(sqrt(2.0 / (1.0 + pow(negativeSlope, 2))))
//         let spatialDimCount = shape.count - 2
//         let receptiveField = shape[0..<spatialDimCount].contiguousSize
//         let fanIn = shape[shape.count - 2] * receptiveField
//         self.init(randomNormal: shape,
//                   stddev: gain / sqrt(Scalar(fanIn)),
//                   generator: &rng
//         )
//     }
// }
// ```

// export
public func leakyRelu<T: TensorFlowFloatingPoint>(
            _ x: Tensor<T>,
            negativeSlope: Double = 0.0
        ) -> Tensor<T> {
    return max(0, x) + T(negativeSlope) * min(0, x)
}
public func gain(_ negativeSlope: Double) -> Double {
    return sqrt(2.0 / (1.0 + pow(negativeSlope, 2.0)))
}

//export
public extension Tensor where Scalar: TensorFlowFloatingPoint {
    init(kaimingUniform shape: TensorShape, negativeSlope: Double = 1.0) {
        // Assumes Leaky ReLU nonlinearity
        let gain = Scalar.init(TensorFlow.sqrt(2.0 / (1.0 + TensorFlow.pow(negativeSlope, 2))))
        let spatialDimCount = shape.count - 2
        let receptiveField = shape[0..<spatialDimCount].contiguousSize
        let fanIn = shape[shape.count - 2] * receptiveField
        let bound = TensorFlow.sqrt(Scalar(3.0)) * gain / TensorFlow.sqrt(Scalar(fanIn))
        self = bound * (2 * Tensor(randomUniform: shape) - 1)
    }
}

public struct Model: Layer {

 public init(){
     
        conv1 = FAConv2D<Float>(
            filterShape: (5, 5, 1, 8),   strides: (2, 2), padding: .same, activation: relu)
        conv2 = FAConv2D<Float>(
            filterShape: (3, 3, 8, 16),  strides: (2, 2), padding: .same, activation: relu)
        conv3 = FAConv2D<Float>(
            filterShape: (3, 3, 16, 32), strides: (2, 2), padding: .same, activation: relu)
        conv4 = FAConv2D<Float>(
            filterShape: (3, 3, 32, 1),  strides: (2, 2), padding: .valid)
    }

    public var conv1 = FAConv2D<Float>(
        filterShape: (5, 5, 1, 8),   strides: (2, 2), padding: .same, activation: relu
    )
    public var conv2 = FAConv2D<Float>(
        filterShape: (3, 3, 8, 16),  strides: (2, 2), padding: .same, activation: relu
    )
    public var conv3 = FAConv2D<Float>(
        filterShape: (3, 3, 16, 32), strides: (2, 2), padding: .same, activation: relu
    )
    public var conv4 = FAConv2D<Float>(
        filterShape: (3, 3, 32, 1),  strides: (2, 2), padding: .valid
    )
    public var flatten = Flatten<Float>()
   
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv1, conv2, conv3, conv4, flatten)
    }
}
// import NotebookExport
// let exporter = NotebookExport(Path.cwd/"02a_why_sqrt5.ipynb")
// print(exporter.export(usingPrefix: "FastaiNotebook_"))