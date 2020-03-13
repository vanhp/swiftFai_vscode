// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "loadData",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "loadData",
            targets: ["loadData"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/mxcl/Path.swift", from: "0.16.1"),
        .package(url: "https://github.com/saeta/Just", from: "0.7.2"),
        .package(url: "https://github.com/latenitesoft/NotebookExport", from: "0.5.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "loadData",
            dependencies: ["Path","Just","NotebookExport"]),
        .testTarget(
            name: "loadDataTests",
            dependencies: ["loadData"]),
    ]
)
