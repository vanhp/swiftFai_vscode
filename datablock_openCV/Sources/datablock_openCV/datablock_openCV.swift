
import Path
import loadData

// 1  Data block foundations

struct datablock_openCV {
    var text = "Hello, datablock_openCV!"
}

//export
public let dataPath = Path.home/".fastai"/"data"

// 1.1  Image ItemList
//export
public func downloadImagenette(path: Path = dataPath) -> Path {
    let url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz"
    let fname = "imagenette-160"
    let file = path/fname
    try! path.mkdir(.p)
    if !file.exists {
        downloadFile(url, dest:(path/"\(fname).tgz").string)
        _ = "/bin/tar".shell("-xzf", (path/"\(fname).tgz").string, "-C", path.string)
    }
    return file
}
