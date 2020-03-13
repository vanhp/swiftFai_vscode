import XCTest

import loadDataTests

var tests = [XCTestCaseEntry]()
tests += loadDataTests.allTests()
XCTMain(tests)
