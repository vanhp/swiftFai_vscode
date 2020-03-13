
import datablock_functional

let path = ImageNette.download()

// Now we compute the next value, the array of all paths.
var allPaths = collectFiles(under: path, recurse: true, filtering: ImageNette.extensions)
// If we look at a random element, compared to the imagenette root, 
// it has the filesystem layout structure we expected
print("description: ",(path.string, allPaths.randomElement()!.string))

// Let us verify that our configurations functions correctly encode 
// the segment (train or val) and the label of an arbitrary item:
print("path sample: ",describeSample(allPaths.randomElement()!))

// We pass the ImageNette.isTraining test function into the partitioner directly
var samples = partitionIntoTrainVal(allPaths, isTrain:ImageNette.isTraining)
describeSample(samples.valid.randomElement()!)
describeSample(samples.train.randomElement()!)

// Let us create a labelNumber mapper from the training data. 
// First we use the function labelOf to get all the training labels, 
// then we can initialize a StringIntMapper.
var trainLabels = samples.train.map(ImageNette.labelOf)
var labelMapper = StringIntMapper(labels: trainLabels)

// The labelMapper now supplies the two bidirectional functions. 
// We can verify they have the required inverse relationship:
var randomLabel = labelMapper.labelMap.randomElement()!
print("label = \(randomLabel)")
var numericalizedLabel = labelMapper.labelToInt(randomLabel)
print("number = \(numericalizedLabel)")
var labelFromNumber = labelMapper.intToLabel(numericalizedLabel)
print("label = \(labelFromNumber)")

// The we define a function which map a raw sample (Path) to a numericalized label (Int)
var pathToNumericalizedLabel = ImageNette.labelOf >| labelMapper.labelToInt

// Now we can, if we wish, compute numericalized labels over all the training and validation items:
var trainNumLabels = samples.train.map(pathToNumericalizedLabel)
var validNumLabels = samples.valid.map(pathToNumericalizedLabel)

let allPaths2 = collectFiles(under: path, recurse: true, filtering: ImageNette.extensions)
let sld = SplitLabeledData(paths: allPaths2)

// // 1.2.6  opening images
// // We can use the same compose approach to convert our images from `Path` filenames to resized images.
// import Foundation
// import SwiftCV
// // First let's open those images with openCV:
// func openImage(_ fn: Path) -> Mat {
//     return imdecode(try! Data(contentsOf: fn.url))
// }
// // And add a convenience function to have a look.
// func showCVImage(_ img: Mat) {
//     let tensImg = Tensor<UInt8>(cvMat: img)!
//     let numpyImg = tensImg.makeNumpyArray()
//     plt.imshow(numpyImg) 
//     plt.axis("off")
//     plt.show()
// }
 showCVImage(openImage(sld.train.randomElement()!.x))
// // The channels are in BGR instead of RGB so we first switch them with openCV
// func BGRToRGB(_ img: Mat) -> Mat {
//     return cvtColor(img, nil, ColorConversionCode.COLOR_BGR2RGB)
// }
// // Then we can resize them
// func resize(_ img: Mat, size: Int) -> Mat {
//     return resize(img, nil, Size(size, size), 0, 0, InterpolationFlag.INTER_LINEAR)
// }

// With our compose operator, the succession of transforms can be written in this pretty way:
let transforms = openImage >| BGRToRGB >| { resize($0, size: 224) }
// And we can have a look at one of our elements:
showCVImage(transforms(sld.train.randomElement()!.x))

// // 1.3  Conversion to Tensor and batching¶
// // Now we will need tensors to train our model, so we need to convert our images and ints to tensors.
// func cvImgToTensorInt(_ img: Mat) -> Tensor<UInt8> {
//     return Tensor<UInt8>(cvMat: img)!
// }
// We compose our transforms with that last function to get tensors.
let pathToTF = transforms >| cvImgToTensorInt
// func intTOTI(_ i: Int) -> TI { return TI(Int32(i)) } 

// // Now we define a `Batcher` that will be responsible for creating minibatches as an iterator. 
// // It has the properties you know from PyTorch (batch size, num workers, shuffle) 
// // and will use multiprocessing to gather the images in parallel.

// // To be able to write `for batch in Batcher(...)`, `Batcher` needs to conform to `Sequence`, 
// // which means it needs to have a `makeIterator` function. That function has 
// // to return another struct that conforms to `IteratorProtocol`. The only thing required 
// // there is a `next` property that returns the next batch (or `nil` if we are finished).

// // The code is pretty striaghtforward: we shuffle the dataset at each beginning of iteration 
// // if we want, then we apply the transforms in parallel with the use of `concurrentMap`, 
// // that works just like map but with `numWorkers` processes.

// struct Batcher: Sequence {
//     let dataset: [(Path, Int)]
//     let xToTensor: (Path) -> Tensor<UInt8>
//     let yToTensor: (Int) ->  TI
//     var bs: Int = 64
//     var numWorkers: Int = 4
//     var shuffle: Bool = false
    
//     init(_ ds: [(Path, Int)], xToTensor: @escaping (Path) -> Tensor<UInt8>, yToTensor: @escaping (Int) ->  TI,
//          bs: Int = 64, numWorkers: Int = 4, shuffle: Bool = false) {
//         (dataset,self.xToTensor,self.yToTensor,self.bs) = (ds,xToTensor,yToTensor,bs)
//         (self.numWorkers,self.shuffle) = (numWorkers,shuffle)
//     }
    
//     func makeIterator() -> BatchIterator { 
//         return BatchIterator(self, numWorkers: numWorkers, shuffle: shuffle)
//     }
// }
// struct BatchIterator: IteratorProtocol {
//     let b: Batcher
//     var numWorkers: Int = 4
//     private var idx: Int = 0
//     private var ds: [(Path, Int)]
    
//     init(_ batcher: Batcher, numWorkers: Int = 4, shuffle: Bool = false){ 
//         (b,self.numWorkers,idx) = (batcher,numWorkers,0) 
//         self.ds = shuffle ? b.dataset.shuffled() : b.dataset
//     }
    
//     mutating func next() -> (xb:TF, yb:TI)? {
//         guard idx < b.dataset.count else { return nil }
//         let end = idx + b.bs < b.dataset.count ? idx + b.bs : b.dataset.count 
//         let samples = Array(ds[idx..<end])
//         idx += b.bs
//         return (xb: TF(Tensor<UInt8>(concatenating: samples.concurrentMap(nthreads: numWorkers) { 
//             self.b.xToTensor($0.0).expandingShape(at: 0) }))/255.0, 
//                 yb: TI(concatenating: samples.concurrentMap(nthreads: numWorkers) { 
//             self.b.yToTensor($0.1).expandingShape(at: 0) }))
//     }
    
// }

SetNumThreads(0)

let batcher = Batcher(sld.train, xToTensor: pathToTF, yToTensor: intTOTI, bs:256, shuffle:true)

time {var c = 0
      for batch in batcher { c += 1 }
     }

let firstBatch = batcher.first(where: {_ in true})!
 showTensorImage(firstBatch.xb[0])

 let batcher1 = Batcher1(sld.train, xToTensor: transforms, collateFunc: collateMats, yToTensor: intTOTI, bs:256, shuffle:true)    

time {var c = 0
      for batch in batcher1 { c += 1 }
     }