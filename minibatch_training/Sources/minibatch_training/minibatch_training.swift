
import TensorFlow
import loadData
import fastai_layers
import fully_connected

struct minibatch_training {
    var text = "Hello, minibatch_training!"
}

// Our labels will be integers from now on, so to go with our TF abbreviation, we introduce TI.
// export

/// short name for lable of type Int32
public typealias TI = Tensor<Int32>

/// model with 2 fully connected layers
public struct MyModel: Layer {
    public var layer1: FADense<Float>
    public var layer2: FADense<Float>
    
    public init(nIn: Int, nHid: Int, nOut: Int){
        layer1 = FADense(nIn, nHid, activation: relu)
        layer2 = FADense(nHid, nOut)
    }
    
    @differentiable
    public func callAsFunction(_ input: TF) -> TF {
        return input.sequenced(through: layer1, layer2)
    }
}

// ### Cross entropy loss
// Before we can train our model, we need to have a loss function. 
// We saw how to write `logSoftMax` from scratch in PyTorch, but let's do it once in swift too.

///log soft max
public func logSoftmax<Scalar>(_ activations: Tensor<Scalar>) -> 
                Tensor<Scalar> where Scalar:TensorFlowFloatingPoint{
    let exped = exp(activations) 
    return log(exped / exped.sum(alongAxes: -1))
}

///negative log likelyhood
public func nll<Scalar>(_ input: Tensor<Scalar>, _ target :TI) -> Tensor<Scalar> 
    where Scalar:TensorFlowFloatingPoint{
        let idx: TI = _Raw.range(start: Tensor(0), 
                                limit: Tensor(numericCast(target.shape[0])), 
                                delta: Tensor(1))
        let indices = _Raw.concat(concatDim: Tensor(1), 
                                [idx.expandingShape(at: 1), 
                                target.expandingShape(at: 1)])
        let losses = _Raw.gatherNd(params: input, indices: indices)
        return -losses.mean()
}

/// Simplify logSoftmax with log formulas.
 public func logSoftmax_V2<Scalar>(_ activations: Tensor<Scalar>) -> 
                Tensor<Scalar> where Scalar:TensorFlowFloatingPoint{
    return activations - log(exp(activations).sum(alongAxes: -1))
}

/// We now use the LogSumExp trick to avoid floating point number diverge
public func logSumExp<Scalar>(_ x: Tensor<Scalar>) -> 
                Tensor<Scalar> where Scalar:TensorFlowFloatingPoint{
    let m = x.max(alongAxes: -1)
    return m + log(exp(x-m).sum(alongAxes: -1))
}

/// log softmax with sum of exponential
public func logSoftmax_SumExp<Scalar>(_ activations: Tensor<Scalar>) -> 
                Tensor<Scalar> where Scalar:TensorFlowFloatingPoint{
    return activations - logSumExp(activations)
}

// ## Basic training loop
// Basically the training loop repeats over the following steps:
// - get the output of the model on a batch of inputs
// - compare the output to the labels we have and compute a loss
// - calculate the gradients of the loss with respect to every parameter of the model
// - update said parameters with those gradients to make them a little bit better

// export

/// accuracy metric
public func accuracy(_ output: TF, _ target: TI) -> TF{
    let corrects = TF(output.argmax(squeezingAxis: 1) .== target)
    return corrects.mean()
}
// Here's a handy function (thanks for Alexis Gallagher) to grab a batch of indices at a time.

//export

///to grab a batch of indices at a time.
public func batchedRanges(start:Int, end:Int, bs:Int) -> UnfoldSequence<Range<Int>,Int>{
  return sequence(state: start) { (batchStart) -> Range<Int>? in
    let remaining = end - batchStart
    guard remaining > 0 else { return nil}
    let currentBs = min(bs,remaining)
    let batchEnd = batchStart.advanced(by: currentBs)
    defer {  batchStart = batchEnd  }
    return batchStart ..< batchEnd
  }
}

// 1.2  Dataset
// We can create a swift Dataset from our arrays. It will automatically batch things for us:

// export

/// grab data by the batch
public struct DataBatch<Inputs: Differentiable & TensorGroup, Labels: TensorGroup>: TensorGroup {
    public var xb: Inputs
    public var yb: Labels
    
    public init(xb: Inputs, yb: Labels){ (self.xb,self.yb) = (xb,yb) }
    
    public var _tensorHandles: [_AnyTensorHandle] {
        xb._tensorHandles + yb._tensorHandles
    }
    
    public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
        let xStart = _handles.startIndex
        let xEnd = _handles.index(xStart, offsetBy: Int(Inputs._tensorHandleCount))
        self.xb = Inputs.init(_handles: _handles[xStart..<xEnd])
        self.yb = Labels.init(_handles: _handles[xEnd..<_handles.endIndex])
    }
}

// 1.2.1  Training loop¶
// With everything before, we can now write a generic training loop. 
// It needs two generic types: the optimizer (Opt) and the labels (Label):

public func train<Opt: Optimizer, Label:TensorGroup>(
                _ model: inout Opt.Model,
                on ds: Dataset<DataBatch<Opt.Model.Input, Label>>,
                using opt: inout Opt,
                lossFunc: @escaping @differentiable (Opt.Model.Output, @noDerivative Label) -> 
                            Tensor<Opt.Scalar> ) where Opt.Model: Layer,
                                                        Opt.Model.Input: TensorGroup,
                                                        Opt.Scalar: TensorFlowFloatingPoint {
    for batch in ds {
        let (loss, 𝛁model) = model.valueWithGradient {
            lossFunc($0(batch.xb), batch.yb)
        }
        opt.update(&model, along: 𝛁model)
    }
}

// We can't use directly sofmaxCrossEntropy because it has a reduction parameter, 
// so we define a fastai version.

//export

@differentiable(wrt: logits)
public func crossEntropy(_ logits: TF, _ labels: TI) -> TF {
    return softmaxCrossEntropy(logits: logits, labels: labels)

}

// import NotebookExport
// let exporter = NotebookExport(Path.cwd/"03_minibatch_training.ipynb")
// print(exporter.export(usingPrefix: "FastaiNotebook_"))