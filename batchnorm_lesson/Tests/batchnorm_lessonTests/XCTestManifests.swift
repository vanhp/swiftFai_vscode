import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(batchnorm_lessonTests.allTests),
    ]
}
#endif
