
import Path
import loadData
import TensorFlow
import callbacks
import minibatch_training
import matmul
import fully_connected
import batchnorm
import fastai_layers

import Python
let plt = Python.import("matplotlib.pyplot")


struct datablock {
    var text = "Hello, datablock!"
}

//export
public func downloadImagenette(path: Path, sz:String="-160") -> Path {
    let url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette\(sz).tgz"
    let fname = "imagenette\(sz)"
    let file = path/fname
    try! path.mkdir(.p)
    if !file.exists {
        downloadFile(url, dest:(path/"\(fname).tgz").string)
        _ = "/usr/bin/env".shell("tar", "-xzf", (path/"\(fname).tgz").string, "-C", path.string)
    }
    return file
}

//export
public func showImg<T:NumpyScalarCompatible>(_ img: Tensor<T>, _ w: Int = 7, _ h: Int = 5) {
    showImg(img.makeNumpyArray(), w, h)
}

public func showImg(_ img: PythonObject, _ w: Int = 7, _ h: Int = 5) {
    plt.figure(figsize: [w, h])
    plt.imshow(img)
    plt.axis("off")
    plt.show()
}

// 1.1.2  Grab all the images
// Now that we have donloaded the data, we need to be able to recursively 
// grab all the filenames in the imagenette folder. 
// The following function walks recursively through 
// the folder and adds the filenames that have the right extension.
//export
public func fetchFiles(path: Path, recurse: Bool = false, extensions: [String]? = nil) -> [Path] {
    var res: [Path] = []
    for p in try! path.ls(){
        if p.kind == .directory && recurse { 
            res += fetchFiles(path: p.path, recurse: recurse, extensions: extensions)
        } else if extensions == nil || extensions!.contains(p.path.extension.lowercased()) {
            res.append(p.path)
        }
    }
    return res
}
// Note that we don't have a generic open_image function like in python here, 
// but will be using a specific decode function (here for jpegs, but there is one for gifs or pngs). 
// That's why we limit ourselves to jpeg exensions here.

// 1.2  Prepare the data
// Dataset can handle all the transforms that go on a Tensor, 
// including opening an image and resizing it since it takes StringTensor. 
// That makes the tfms attribute of ItemList irrelevant, 
// so ItemList is just an array of Item with a path (if get method seems useful later, we can add it).

// export
public struct ItemList<Item>{
    public var items: [Item]
    public let path: Path
    
    public init(items: [Item], path: Path){
        (self.items,self.path) = (items,path)
    }
}
// export
public extension ItemList where Item == Path {
    init(fromFolder path: Path, extensions: [String], recurse: Bool = true) {
        self.init(items: fetchFiles(path: path, recurse: recurse, extensions: extensions), path:  path)
    }
}

///////////////////////////////////////////
// Data preparation

// split data
// data into training set and validation set

// export
public struct SplitData<Item>{
    public let train: ItemList<Item>
    public let valid: ItemList<Item>
    public var path: Path { return train.path }
    
    public init(train: ItemList<Item>, valid: ItemList<Item>){
        (self.train, self.valid) = (train, valid)
    }
    
    public init(_ il: ItemList<Item>, fromFunc: (Item) -> Bool){
        self.init(train: ItemList(items: il.items.filter { !fromFunc($0) }, path: il.path),
                  valid: ItemList(items: il.items.filter {  fromFunc($0) }, path: il.path))
    }
}

// export
public func grandParentSplitter(fName: Path, valid: String = "valid") -> Bool{
    return fName.parent.parent.basename() == valid
}

// Processor

// export
public protocol Processor {
    associatedtype Input
    associatedtype Output
    
    mutating func initState(items: [Input])
    func process1(item: Input) -> Output
    func deprocess1(item: Output) -> Input
}

// export
public extension Processor {
    func process(items: [Input]) -> [Output] {
        return items.map { process1(item: $0) }
    }    
    func deprocess(items: [Output]) -> [Input] {
        return items.map { deprocess1(item: $0) }
    }
}

// export
public struct NoopProcessor<Item>: Processor {
    public init() {}
    public mutating func initState(items: [Item]) {}
    public func process1  (item: Item) -> Item { return item }
    public func deprocess1(item: Item) -> Item { return item }
}

// export
public struct CategoryProcessor: Processor {
    public init() {}
    public var vocab: [String]? = nil
    public var reverseMap: [String: Int32]? = nil
    
    public mutating func initState(items: [String]) {
        vocab = Array(Set(items)).sorted()
        reverseMap = [:]
        for (i,x) in vocab!.enumerated() { reverseMap![x] = Int32(i) }
    }
    
    public func process1  (item: String) -> Int32 { return reverseMap![item]! }
    public func deprocess1(item: Int32)  -> String { return vocab![Int(item)] }
}

// Label

// When we build the datasets, we don't need to return a tuple (item, label) 
// but to have the tensor(s) with the items and the tensor(s) with the labels separately.

//export

/// Put label on the item List
public struct LabeledItemList<PI,PL> where PI: Processor, PL: Processor{
    public var items: [PI.Output]
    public var labels: [PL.Output]
    public let path: Path
    public var procItem: PI
    public var procLabel: PL
    
    public init(rawItems: [PI.Input], rawLabels: [PL.Input], path: Path, procItem: PI, procLabel: PL){
        (self.procItem,self.procLabel,self.path) = (procItem,procLabel,path)
        self.items = procItem.process(items: rawItems)
        self.labels = procLabel.process(items: rawLabels)
    }
    
    public init(_ il: ItemList<PI.Input>, fromFunc: (PI.Input) -> PL.Input, procItem: PI, procLabel: PL){
        self.init(            
                    rawItems:  il.items,
                    rawLabels: il.items.map{ fromFunc($0)},
                    path:      il.path,
                    procItem:  procItem,
                    procLabel: procLabel                 
                  )
    }
    
    public func rawItem (_ idx: Int) -> PI.Input { return procItem.deprocess1 (item: items[idx])  }
    public func rawLabel(_ idx: Int) -> PL.Input { return procLabel.deprocess1(item: labels[idx]) }
}

//export

/// split labelled data into training label and validation labels list
public struct SplitLabeledData<PI,PL> where PI: Processor, PL: Processor{
    public let train: LabeledItemList<PI,PL>
    public let valid: LabeledItemList<PI,PL>
    public var path: Path { return train.path }
    
    public init(train: LabeledItemList<PI,PL>, valid: LabeledItemList<PI,PL>){
        (self.train, self.valid) = (train, valid)
    }
    
    public init(_ sd: SplitData<PI.Input>, 
                fromFunc: (PI.Input) -> PL.Input, 
                procItem: inout PI, 
                procLabel: inout PL){
        procItem.initState(items: sd.train.items)
        let trainLabels = sd.train.items.map{ fromFunc($0) }
        procLabel.initState(items: trainLabels)
        self.init(train: LabeledItemList(rawItems: sd.train.items, 
                                         rawLabels: trainLabels, 
                                         path: sd.path, 
                                         procItem: procItem, 
                                         procLabel: procLabel),
                  valid: LabeledItemList(sd.valid, fromFunc: fromFunc, procItem: procItem, procLabel: procLabel))
    }
}

/// Make a labeled data without an input processor, by defaulting to a noop processor.
public func makeLabeledData<T, PL: Processor>(_ sd: SplitData<T>, 
                                              fromFunc: (T) -> PL.Input, 
                                              procLabel: inout PL) -> SplitLabeledData<NoopProcessor<T>, PL> {
    var pi = NoopProcessor<T>()
    return SplitLabeledData(sd, fromFunc: fromFunc, procItem: &pi, procLabel: &procLabel)
}

//export

public func parentLabeler(_ fName: Path) -> String { return fName.parent.basename() }

// 1.2.4  Datasets
// To go in a Dataset, our array of items and array of labels need to be converted to tensors.

// export
public struct LabeledElement<I: TensorGroup, L: TensorGroup> : TensorGroup {
    public var xb: I
    public var yb: L    
    
    public init(xb: I, yb: L) {
        (self.xb, self.yb) = (xb, yb)
    }
    
    // Explicit implementation to make this struct work well with LazyTensor.
    // These will be derived automatically in the future.
    public var _tensorHandles: [_AnyTensorHandle] {
        xb._tensorHandles + yb._tensorHandles
    }
    
    public init<C: RandomAccessCollection>( _handles: C) where C.Element : _AnyTensorHandle {
        let xStart = _handles.startIndex
        let xEnd = _handles.index(
                        xStart, offsetBy: Int(I._tensorHandleCount))
                        self.xb = I.init(_handles: _handles[xStart..<xEnd])
                        self.yb = L.init(_handles: _handles[xEnd..<_handles.endIndex])
    }
}

// export
public extension SplitLabeledData {
    func toDataBunch<XB, YB> (
                                itemToTensor: ([PI.Output]) -> XB,
                                labelToTensor: ([PL.Output]) -> YB, 
                                bs: Int = 64) 
                                    -> DataBunch<LabeledElement<XB, YB>> where XB: TensorGroup, YB: TensorGroup {
        let trainDs = Dataset<LabeledElement<XB, YB>>(
                                                        elements: LabeledElement( xb: itemToTensor(train.items), 
                                                                                  yb: labelToTensor(train.labels)))
        let validDs = Dataset<LabeledElement<XB, YB>>(
                                                        elements: LabeledElement(xb: itemToTensor(valid.items), 
                                                                                 yb: labelToTensor(valid.labels)))
        return DataBunch(train: trainDs, 
                         valid: validDs, 
                         trainLen: train.items.count, 
                         validLen: valid.items.count,
                         bs: bs)
    }
}

// export

public func pathsToTensor(_ paths: [Path]) -> StringTensor { return StringTensor(paths.map{ $0.string })}
public func intsToTensor(_ items: [Int32]) -> Tensor<Int32> { return Tensor<Int32>(items)}

// 1.2.5  Transforms
// We directly plug in to the dataset the transforms we want to apply.

// export

public func transformData<I,TI,L>(
                                    _ data: DataBunch<LabeledElement<I,L>>, 
                                    nWorkers:Int=4,
                                    tfmItem: (I) -> TI) 
                                        -> DataBunch<DataBatch<TI,L>> where I: TensorGroup, 
                                                                            TI: TensorGroup & Differentiable, 
                                                                            L: TensorGroup {
            //             var x =  data.train.innerDs.map(parallelCallCount: nWorkers){  //crash here!
            //             var xxb = tfmItem($0.xb)
            //             var yyb = $0.yb
            //                 DataBatch(xb: xxb, 
            //                 yb: yyb)}           
            // return DataBunch(train: x ,
            //          valid: data.valid.innerDs.map(parallelCallCount: nWorkers){ DataBatch(xb: tfmItem($0.xb), yb: $0.yb) },
            //          trainLen: data.train.dsCount, 
            //          validLen: data.valid.dsCount,
            //          bs: data.train.bs)                                      
     return DataBunch(train: data.train.innerDs.map(parallelCallCount: nWorkers){ DataBatch(xb: tfmItem($0.xb), yb: $0.yb) },
                      valid: data.valid.innerDs.map(parallelCallCount: nWorkers){ DataBatch(xb: tfmItem($0.xb), yb: $0.yb) },
                      trainLen: data.train.dsCount, 
                      validLen: data.valid.dsCount,
                      bs: data.train.bs)
}

// export

public func openAndResize(fname: StringTensor, size: Int) -> TF{
    // crash here why calling StringTensor to decodeJpeg function to decode fname
    let decodedImg = StringTensor(readFile: fname).decodeJpeg(channels: 3)
    let resizedImg = Tensor<Float>(_Raw.resizeBilinear(
                            images: Tensor<UInt8>([decodedImg]), 
                            size: Tensor<Int32>([Int32(size), Int32(size)]))) / 255.0
    return resizedImg.reshaped(to: TensorShape(size, size, 3))
}
// export
public extension FADataset {
    func oneBatch() -> Element? {
        for batch in ds { return batch }
        return nil
    }
} 

// export
public func showImages(_ xb: TF, labels: [String]? = nil) {
    let (rows,cols) = (3,3)
    plt.figure(figsize: [9, 9])
    for i in 0..<(rows * cols) {
        let img = plt.subplot(rows, cols, i + 1)
        img.axis("off")
        let x = xb[i].makeNumpyArray()
        img.imshow(x)
        if labels != nil { img.set_title(labels![i]) }
        if (i + 1) >= (rows * cols) { break }
    }
    plt.show()
}

//export 
public let imagenetStats = (mean: TF([0.485, 0.456, 0.406]), std: TF([0.229, 0.224, 0.225]))

//export
public func prevPow2(_ x: Int) -> Int { 
    var res = 1
    while res <= x { res *= 2 }
    return res / 2
}

//export
public struct CNNModel: Layer {
    public var convs: [ConvBN<Float>]
    public var pool = FAGlobalAvgPool2D<Float>()
    public var linear: FADense<Float>
    
    public init(channelIn: Int, nOut: Int, filters: [Int]){
        convs = []
        let (l1,l2) = (channelIn, prevPow2(channelIn * 9))
        convs = [ConvBN(l1,   l2,   stride: 1),
                 ConvBN(l2,   l2*2, stride: 2),
                 ConvBN(l2*2, l2*4, stride: 2)]
        let allFilters = [l2*4] + filters
        for i in 0..<filters.count { convs.append(ConvBN(allFilters[i], allFilters[i+1], stride: 2)) }
        linear = FADense<Float>(filters.last!, nOut)
    }
    
    @differentiable
    public func callAsFunction(_ input: TF) -> TF {
        return linear(pool(convs(input)))
    }
}


// import NotebookExport
// let exporter = NotebookExport(Path.cwd/"08_data_block.ipynb")
// print(exporter.export(usingPrefix: "FastaiNotebook_"))