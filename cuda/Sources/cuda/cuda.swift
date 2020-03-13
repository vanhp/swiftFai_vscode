
import TensorFlow
import fastai_layers
import callbacks
import fully_connected




struct cuda {
    var text = "Hello, cuda!"
}

// export
extension Learner {
    public class AddChannel: Delegate {
        public override func batchWillStart(learner: Learner) {
            learner.currentInput = learner.currentInput!.expandingShape(at: -1)
        }
    }
    
    public func makeAddChannel() -> AddChannel { return AddChannel() }
}

//export 
public struct CnnModel: Layer {
    public var convs: [FAConv2D<Float>]
    public var pool = FAGlobalAvgPool2D<Float>()
    public var linear: FADense<Float>
    
    public init(channelIn: Int, nOut: Int, filters: [Int]){
        let allFilters = [channelIn] + filters
        convs = Array(0..<filters.count).map { i in
            return FAConv2D(allFilters[i], allFilters[i+1], ks: 3, stride: 2)
        }
        linear = FADense<Float>(filters.last!, nOut)
    }
    
    @differentiable
    public func callAsFunction(_ input: TF) -> TF {
        return linear(pool(convs(input)))
    }
}

// 3  Collect Layer Activation Statistics
public class ActivationStatsHook {
    public private (set) var means: [Float] = []
    public private (set) var stds: [Float] = []    
   public func update(_ act: TF) {
        means.append(act.mean().scalarized())
        stds.append (act.std() .scalarized())
    }
    public init(){
        means = []
        stds = []
    }

}

// import NotebookExport
// let exporter = NotebookExport(Path.cwd/"06_cuda.ipynb")
// print(exporter.export(usingPrefix: "FastaiNotebook_"))