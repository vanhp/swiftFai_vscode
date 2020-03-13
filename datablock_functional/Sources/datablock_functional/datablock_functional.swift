
import Path
import loadData


struct datablock_functional {
    var text = "Hello, datablock_functional!"
}

//1  Data block foundations, in Swifty/functional style
// The DataBlock API in Python is designed to help with the routine data manipulations 
// involved in modelling: downloading data, 
// loading it given an understanding of its layout on the filesystem, processing it, 
// and feeding it into an ML framework like fastai. This is a data pipeline. How do we do this in Swift?

// One approach is to build a set of types (structs, protocols, etc.) 
// which represent various stages of this pipeline. By making the types generic, 
// we could build a library that handled data for many kinds of models. However, 
// it is sometimes a good rule of thumb, before writing generic types, 
// to start by writing concrete types and then to notice 
// what to abstract into a generic later. And another good rule of thumb, 
// before writing concrete types, is to write no types at all, 
// and to see how far you can get with a more primitive tool for composition: functions.

// This notebook shows how to perform DataBlock-like operations 
// using a _lightweight functional style_. This means, first, to rely as much as 
// possible on _pure_ functions -- that is, functions which do nothing 
// but return outputs based on their inputs, and which don't 
// mutate values anywhere. Second, in particular, it means to use 
// Swift's support for _higher-order functions_ 
// (functions which take functions, like `map`, `filter`, `reduce`, and `compose`). 
// Finally, this example relies on _tuples_. Like structs, tuples can have named, 
// typed properties. Unlike structs, you don't need to name them. 
// They can be a fast, ad-hoc way to explore the data types that you actually need, 
// without being distracted by considering what's a method, an initializer, etc.,

// Swift has excellent, understated support for a such a style. 

public let dataPath = Path.home/".fastai"/"data"

public func downloadImagenette(path: Path = dataPath, sz:String="-320") -> Path {
    let url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette\(sz).tgz"
    let fname = "imagenette\(sz)"
    let file = path/fname
    try! path.mkdir(.p)
    if !file.exists {
        downloadFile(url, dest:(path/"\(fname).tgz").string)
        _ = "/bin/tar".shell("-xzf", (path/"\(fname).tgz").string, "-C", path.string)
    }
    return file
}

/**
 * Here is what we know ahead of time about how imagenette data is laid out on disk:

```
.
└── data                                           # <-- this is the fastai data root path
    ├── imagenette-160                             # <-- this is the imagenette dataset path
    │   ├── train                                  # <-- the train/ and val/ dirs are our two segments
    │   │   ├── n01440764                          # <-- this is an image category _label_
    │   │   │   ├── n01440764_10026.JPEG           # <-- this is an image (a _sample_) with that label
    │   │   │   ├── n01440764_10027.JPEG
    │   │   │   ├── n01440764_10042.JPEG
   ...
    │   ├── val
    │       └── n03888257
    │           ├── ILSVRC2012_val_00001440.JPEG
    │           ├── ILSVRC2012_val_00002508.JPEG
   ...  

```

We will define one type, an `enum`, to capture this information.

This "empty" `enum` will serve only as a namespace, a grouping, for pure functions 
representing this information. By putting this information into one type, 
our code is more modular: it more clearly distinguishes facts about _this dataset_, 
from _general purpose data manipulators_, from _computations for this analysis_.
 */

// export

public enum ImageNette
{
  /// Downloads imagenette given the fastai data, and returns its "dataset root"
 public  static func download() -> Path { return downloadImagenette() }

  /// Extensions of paths which represent imagenette samples (i.e., items)
  public static var extensions:[String] = ["jpeg", "jpg"]

  // Returns whether an image is in the training set (vs validation set), based on its path
  public static func isTraining(_ p:Path) -> Bool {
    return p.parent.parent.basename() == "train"
  }

  /// Returns an image's label given the image's path
  public static func labelOf(_ p:Path) -> String { return p.parent.basename() }
}

// After download, this `path` is the first _value_, and the remaining steps of 
// analysis can all be seen as applying functions which successively compute new values from past values.

// For instance, this path and the allowed files extensions are the _input_ 
// to the function `collectFilePaths(under:filteringToExtensions)` 
// which _outputs_ an array of all samples in the dataet. (This is like `fetchFiles` 
// but we rename it here to to emphasize the conventional functional 
// operation of "filtering" and to reflect that it does not actually fetch files over the network:

//export

// traverse the path of directory to collecting files into array filter by their extensions
public func collectFiles(under path: Path, recurse: Bool = false, filtering extensions: [String]? = nil) -> [Path] {
    var res: [Path] = []
    for p in try! path.ls(){
        if p.kind == .directory && recurse { 
            res += collectFiles(under: p.path, recurse: recurse, filtering: extensions)
        } else if extensions == nil || extensions!.contains(p.path.extension.lowercased()) {
            res.append(p.path)
        }
    }
    return res
}

public func describeSample(_ path:Path) 
{
    let isTraining = ImageNette.isTraining(path)
    let label = ImageNette.labelOf(path)
    print("""
          path: \(path.string)
          training?:  \(isTraining)
          label: \(label)
          """)
}

// split data
// Now we want to split our samples into a training and validation sets. 
// Since this is so routine we define a standard function that does so.

// It is enough to take an array and returns a named tuple of two arrays, 
// one for training and one for validation.

// export

/// takes a [T] of items, and returns a tuple (train:[T],val:[T]) items
public func partitionIntoTrainVal<T>(_ items:[T],isTrain:((T)->Bool)) -> (train:[T],valid:[T]){
    return (train: items.filter(isTrain), valid: items.filter { !isTrain($0) })
}
// 1.2.4  Process the data¶
// We process the data by taking all training labels, uniquing them, 
//sorting them, and then defining an integer to represent the label.

// Those numerical labels let us define two functions, a function for 
//label->number and the inverse function number->label.

// But notable point is that the process that produces those functions _is also a function_: 
//the input is a list of training labels, and the output is the label<->number bidirectional mappings.

// That function which creates the bidirectional mapping is just 
//the initializer of the String<->Int mapper we define below:

// export
/// Defines bidirectional maps from String <-> Int32, initialized from a collection of Strings
/// using set to uniquing them and dictionary to convert string to int
/// and array for int to string
public struct StringIntMapper {
  private(set) public var labelMap:[String]
  private(set) public var inverseLabelMap:[String:Int]
  public init<S:Sequence>(labels ls:S) where S.Element == String {
    labelMap = Array(Set(ls)).sorted()
    inverseLabelMap = Dictionary(uniqueKeysWithValues:
      labelMap.enumerated().map({ ($0.element, $0.offset) }))
  }
  public func labelToInt(_ label:String) -> Int { return inverseLabelMap[label]! }
  public func intToLabel(_ labelIndex:Int) -> String { return labelMap[labelIndex] }
}

// Labelling the data
// Now we are in a position to give the data numerical labels.

// Now in order to map from a sample item (a `Path`), to a numerical label (an `Int32)`, 
// we just compose our Path->label function with a label->int function. Curiously, 
// Swift does not define its own compose function, so we defined a `compose` operator `>|` ourselves. 
// We can use it to create our new function as a composition explicitly:

// export
/// define a conpose operator combine multiple functions into new function
infix operator >| 
public func >| <A, B, C>(_ f: @escaping (A) -> B,
                   _ g: @escaping (B) -> C) -> (A) -> C {
    return { g(f($0)) }
}

// We've gotten pretty far just using mostly just variables, functions, 
// and function composition. But one downside is that our results are now scattered 
// over a few different variables, samples, trainNumLabels, valNumLabels. 
// We collect these values into one structure for convenience:
public struct SplitLabeledData {
    var mapper: StringIntMapper
    var train: [(x: Path, y: Int)]
    var valid: [(x: Path, y: Int)]
    
    public init(mapper: StringIntMapper, train: [(x: Path, y: Int)], valid: [(x: Path, y: Int)]) {
        (self.mapper,self.train,self.valid) = (mapper,train,valid)
    }
}

// And we can define a convenience init to build this directly from the filenames.

extension SplitLabeledData {
   public init(paths:[Path]){
        let samples = partitionIntoTrainVal(paths, isTrain:ImageNette.isTraining)
        let trainLabels = samples.train.map(ImageNette.labelOf)
        var labelMapper = StringIntMapper(labels:trainLabels)
        let pathToNumericalizedLabel = ImageNette.labelOf >| labelMapper.labelToInt
        self.init(mapper: labelMapper,
                  train:  samples.train.map { ($0, pathToNumericalizedLabel($0)) },
                  valid:  samples.valid.map { ($0, pathToNumericalizedLabel($0)) })
    }
}

//1.2.6  opening images
// We can use the same compose approach to convert our images from `Path` filenames to resized images.
import Foundation
import SwiftCV
// First let's open those images with openCV:
public func openImage(_ fn: Path) -> Mat {
    return imdecode(try! Data(contentsOf: fn.url))
}
// And add a convenience function to have a look.
public func showCVImage(_ img: Mat) {
    let tensImg = Tensor<UInt8>(cvMat: img)!
    let numpyImg = tensImg.makeNumpyArray()
    plt.imshow(numpyImg) 
    plt.axis("off")
    plt.show()
}
//showCVImage(openImage(sld.train.randomElement()!.x))
// The channels are in BGR instead of RGB so we first switch them with openCV
public func BGRToRGB(_ img: Mat) -> Mat {
    return cvtColor(img, nil, ColorConversionCode.COLOR_BGR2RGB)
}
// Then we can resize them
public func resize(_ img: Mat, size: Int) -> Mat {
    return resize(img, nil, Size(size, size), 0, 0, InterpolationFlag.INTER_LINEAR)
}

//1.3  Conversion to Tensor and batching¶
// Now we will need tensors to train our model, so we need to convert our images and ints to tensors.
public func cvImgToTensorInt(_ img: Mat) -> Tensor<UInt8> {
    return Tensor<UInt8>(cvMat: img)!
}

public func intTOTI(_ i: Int) -> TI { return TI(Int32(i)) } 

// Now we define a `Batcher` that will be responsible for creating minibatches as an iterator. 
// It has the properties you know from PyTorch (batch size, num workers, shuffle) 
// and will use multiprocessing to gather the images in parallel.

// To be able to write `for batch in Batcher(...)`, `Batcher` needs to conform to `Sequence`, 
// which means it needs to have a `makeIterator` function. That function has 
// to return another struct that conforms to `IteratorProtocol`. The only thing required 
// there is a `next` property that returns the next batch (or `nil` if we are finished).

// The code is pretty striaghtforward: we shuffle the dataset at each beginning of iteration 
// if we want, then we apply the transforms in parallel with the use of `concurrentMap`, 
// that works just like map but with `numWorkers` processes.

public struct Batcher: Sequence {
    let dataset: [(Path, Int)]
    let xToTensor: (Path) -> Tensor<UInt8>
    let yToTensor: (Int) ->  TI
    var bs: Int = 64
    var numWorkers: Int = 4
    var shuffle: Bool = false
    
    public init(_ ds: [(Path, Int)], xToTensor: @escaping (Path) -> Tensor<UInt8>, yToTensor: @escaping (Int) ->  TI,
         bs: Int = 64, numWorkers: Int = 4, shuffle: Bool = false) {
        (dataset,self.xToTensor,self.yToTensor,self.bs) = (ds,xToTensor,yToTensor,bs)
        (self.numWorkers,self.shuffle) = (numWorkers,shuffle)
    }
    
   public func makeIterator() -> BatchIterator { 
        return BatchIterator(self, numWorkers: numWorkers, shuffle: shuffle)
    }
}
public struct BatchIterator: IteratorProtocol {
    let b: Batcher
    var numWorkers: Int = 4
    private var idx: Int = 0
    private var ds: [(Path, Int)]
    
    public init(_ batcher: Batcher, numWorkers: Int = 4, shuffle: Bool = false){ 
        (b,self.numWorkers,idx) = (batcher,numWorkers,0) 
        self.ds = shuffle ? b.dataset.shuffled() : b.dataset
    }
    
    public mutating func next() -> (xb:TF, yb:TI)? {
        guard idx < b.dataset.count else { return nil }
        let end = idx + b.bs < b.dataset.count ? idx + b.bs : b.dataset.count 
        let samples = Array(ds[idx..<end])
        idx += b.bs
        return (xb: TF(Tensor<UInt8>(concatenating: samples.concurrentMap(nthreads: numWorkers) { 
            self.b.xToTensor($0.0).expandingShape(at: 0) }))/255.0, 
                yb: TI(concatenating: samples.concurrentMap(nthreads: numWorkers) { 
            self.b.yToTensor($0.1).expandingShape(at: 0) }))
    }
    
}

public func showTensorImage(_ img: TF) {
    let numpyImg = img.makeNumpyArray()
    plt.imshow(numpyImg) 
    plt.axis("off")
    plt.show()
}

// 1.3.1  With fast collate
public protocol Countable {
    var count:Int {get}
}
extension Mat  :Countable {}
extension Array:Countable {}

public extension Sequence where Element:Countable {
    var totalCount:Int { return map{ $0.count }.reduce(0, +) }
}

public func collateMats(_ imgs:[Mat]) -> TF {
    let c = imgs.totalCount
    let ptr = UnsafeMutableRawPointer.allocate(byteCount: c, alignment: 1)
    defer {ptr.deallocate()}
    var p = ptr
    for img in imgs {
        p.copyMemory(from: img.dataPtr, byteCount: img.count)
        p += img.count
    }
    let r = UnsafeBufferPointer(start: ptr.bindMemory(to: UInt8.self, capacity: c), count: c)
    let cvImg = imgs[0]
    let shape = TensorShape([imgs.count, cvImg.rows, cvImg.cols, cvImg.channels])
    let res = Tensor(shape: shape, scalars: r)
    return Tensor<Float>(res)/255.0
}

public struct Batcher1: Sequence {
    let dataset: [(Path, Int)]
    let xToTensor: (Path) -> Mat
    let collateFunc: ([Mat]) -> TF
    let yToTensor: (Int) ->  TI
    var bs: Int = 64
    var numWorkers: Int = 4
    var shuffle: Bool = false
    
    init(_ ds: [(Path, Int)], xToTensor: @escaping (Path) -> Mat, collateFunc: @escaping ([Mat]) -> TF, 
         yToTensor: @escaping (Int) ->  TI,
         bs: Int = 64, numWorkers: Int = 4, shuffle: Bool = false) {
        (dataset,self.xToTensor,self.collateFunc,self.yToTensor) = (ds,xToTensor,collateFunc,yToTensor)
        (self.bs,self.numWorkers,self.shuffle) = (bs,numWorkers,shuffle)
    }
    
    func makeIterator() -> BatchIterator1 { 
        return BatchIterator1(self, numWorkers: numWorkers, shuffle: shuffle)
    }
}

public struct BatchIterator1: IteratorProtocol {
    let b: Batcher1
    var numWorkers: Int = 4
    private var idx: Int = 0
    private var ds: [(Path, Int)]
    
    init(_ batcher: Batcher1, numWorkers: Int = 4, shuffle: Bool = false){ 
        (b,self.numWorkers,idx) = (batcher,numWorkers,0) 
        self.ds = shuffle ? b.dataset.shuffled() : b.dataset
    }
    
    mutating func next() -> (xb:TF, yb:TI)? {
        guard idx < b.dataset.count else { return nil }
        let end = idx + b.bs < b.dataset.count ? idx + b.bs : b.dataset.count 
        let samples = Array(ds[idx..<end])
        idx += b.bs
        return (xb: b.collateFunc(samples.concurrentMap(nthreads: numWorkers) { 
            self.b.xToTensor($0.0) }), 
                yb: TI(concatenating: samples.concurrentMap(nthreads: numWorkers) { 
            self.b.yToTensor($0.1).expandingShape(at: 0) }))
    }
    
}





