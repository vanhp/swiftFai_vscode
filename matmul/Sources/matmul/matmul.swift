
// import loadData
// import Path
 import TensorFlow


struct matmul {
    var text = "Hello, matmul!"
}

// Ok, now that we know how floating point types and arrays work, 
// we can finally build our own matmul from scratch, 
// using a few loops. We will take the two input matrices as single dimensional arrays 
// so we can show manual indexing into them, the hard way:
// a and b are the flattened array elements, aDims/bDims are the #rows/columns of the arrays.
public func swiftMatmul(a: [Float], b: [Float], aDims: (Int,Int), bDims: (Int,Int)) -> [Float] {
    assert(aDims.1 == bDims.0, "matmul shape mismatch")
    
    var res = Array(repeating: Float(0.0), count: aDims.0 * bDims.1)
    for i in 0 ..< aDims.0 {
        for j in 0 ..< bDims.1 {
            for k in 0 ..< aDims.1 {
                res[i*bDims.1+j] += a[i*aDims.1+k] * b[k*bDims.1+j]
            }
        }
    }
    return res
}

// a and b are the flattened array elements, aDims/bDims are the #rows/columns of the arrays.
public func swiftMatmulUnsafe(a: UnsafePointer<Float>, b: UnsafePointer<Float>, aDims: (Int,Int), bDims: (Int,Int)) -> [Float] {
    assert(aDims.1 == bDims.0, "matmul shape mismatch")
    
    var res = Array(repeating: Float(0.0), count: aDims.0 * bDims.1)
    res.withUnsafeMutableBufferPointer { res in 
        for i in 0 ..< aDims.0 {
            for j in 0 ..< bDims.1 {
                for k in 0 ..< aDims.1 {
                    res[i*bDims.1+j] += a[i*aDims.1+k] * b[k*bDims.1+j]
                }
            }
        }
    }
    return res
}

//export
public extension StringTensor {
    // Read a file into a Tensor.
    init(readFile filename: String) {
        self.init(readFile: StringTensor(filename))
    }
    init(readFile filename: StringTensor) {
        self = _Raw.readFile(filename: filename)
    }

    // Decode a StringTensor holding a JPEG file into a Tensor<UInt8>.
    // crash while try to decode the Jpeg image
    func decodeJpeg(channels: Int = 0) -> Tensor<UInt8> {
        return _Raw.decodeJpeg(contents: self, channels: Int64(channels), dctMethod: "") 
    }
}

// import NotebookExport
// let exporter = NotebookExport(Path.cwd/"01_matmul.ipynb")
// print(exporter.export(usingPrefix: "FastaiNotebook_"))