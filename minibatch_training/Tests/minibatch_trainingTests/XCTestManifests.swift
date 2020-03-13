import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(minibatch_trainingTests.allTests),
    ]
}
#endif
