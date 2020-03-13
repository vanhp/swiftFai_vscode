// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swiftfai_Test",
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(path:"../loadData"),
        .package(path:"../matmul"),
        .package(path:"../fastai_layers"),
        .package(path:"../fully_connected"),
        .package(path:"../why_sqrt5"),
        .package(path:"../good_init"),
        .package(path:"../autoDiff_swift"),
        .package(path:"../minibatch_training"),
        .package(path:"../callbacks"),
        .package(path:"../annealing"),
        .package(path:"../early_stoping"),
        .package(path:"../cuda"),
        .package(path:"../batchnorm"),
        .package(path:"../batchnorm_lesson"),
        .package(path:"../datablock"),
        .package(path:"../hetDict"),
        .package(path:"../datablock_openCV"),
        .package(path:"../SwiftSox"),
        //.package(path:"../SwiftCV"),
        .package(path:"../datablock_functional")

    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "swiftfai_Test",
            dependencies: ["loadData","matmul","fastai_layers","fully_connected",
            "why_sqrt5","good_init","autoDiff_swift",
            "minibatch_training","callbacks","annealing",
            "early_stoping","cuda","batchnorm","batchnorm_lesson",
            "datablock","hetDict","datablock_openCV","SwiftSox",
            //"SwiftCV",
            "datablock_functional"
            ]),
        .testTarget(
            name: "swiftfai_TestTests",
            dependencies: ["swiftfai_Test"]),
    ]
)
