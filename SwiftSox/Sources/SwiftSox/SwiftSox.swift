
import Foundation
import sox


import Python
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
// let display = Python.import("IPython.display")
// IPythonDisplay.shell.enable_matplotlib("inline")


struct SwiftSox {
    var text = "Hello, SwiftSox!"
}

// extension String {
//     func toUnsafePointer() -> UnsafePointer<UInt8>? {
//         guard let data = self.data(using: .utf8) else {
//             return nil
//         }

//         let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
//         let stream = OutputStream(toBuffer: buffer, capacity: data.count)
//         stream.open()
//         let value = data.withUnsafeBytes {
//             $0.baseAddress?.assumingMemoryBound(to: UInt8.self)
//         }
//         guard let val = value else {
//             return nil
//         }
//         stream.write(val, maxLength: data.count)
//         stream.close()

//         return UnsafePointer<UInt8>(buffer)
//     }

//     func toUnsafeMutablePointer() -> UnsafeMutablePointer<Int8>? {
//         return strdup(self)
//     }
//      func toUnsafeMutablePointerSox() -> UnsafeMutablePointer<sox_format_t>? {
        
//         return strdup(self)
//     }
// }


public func InitSox() {
  if sox_format_init() != SOX_SUCCESS.rawValue { fatalError("Can not init SOX!") }
}
public func ReadSoxAudio(_ name:String)->UnsafeMutablePointer<sox_format_t> {
      
  return sox_open_read(name,nil, nil, nil)
}

public func readAudio() ->(){
  let fd = ReadSoxAudio("../SwiftSox/sounds/chris.mp3")
  let sig = fd.pointee.signal
  print(sig.rate,sig.precision,sig.channels,sig.length)
  var samples = [Int32](repeating: 0, count: numericCast(sig.length))
  sox_read(fd, &samples, numericCast(sig.length))

  let t = samples.makeNumpyArray()
  plt.figure(figsize: [12, 4])
  plt.plot(t[2000..<4000])
  plt.show()
}