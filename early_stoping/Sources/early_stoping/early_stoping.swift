import TensorFlow
import minibatch_training
import callbacks
import Python
import callbacks
import annealing
import fully_connected


struct early_stoping {
    var text = "Hello, early_stoping!"
}


// Make an extension to quickly load them.
//TODO: when recorder can be accessed as a property, remove it from the return
extension Learner where Opt.Scalar: PythonConvertible {
    public func makeDefaultDelegates(metrics: [(Output, Label) -> TF] = []) -> Recorder {
        let recorder = makeRecorder()
        delegates = [makeTrainEvalDelegate(), makeShowProgress(), recorder]
        if !metrics.isEmpty { delegates.append(makeAvgMetric(metrics: metrics)) }
        return recorder
    }
}

// Control flow
extension Learner {
    public class TestControlFlow: Delegate {
        public override var order: Int { return 3 }
        
        var skipAfter,stopAfter: Int
        public init(skipAfter:Int, stopAfter: Int){(self.skipAfter,self.stopAfter) = (skipAfter,stopAfter) }
        
        public override func batchWillStart(learner: Learner) throws {
            print("batchWillStart")
            if learner.currentIter >= stopAfter {
                throw LearnerAction.stop(reason: "*** stopped: \(learner.currentIter)")
            }
            if learner.currentIter >= skipAfter {
                throw LearnerAction.skipBatch(reason: "*** skipBatch: \(learner.currentIter)")
            }
        }
        
        public override func trainingDidFinish(learner: Learner) {
            print("trainingDidFinish")
        }
        
        public override func batchSkipped(learner: Learner, reason: String) {
            print(reason)
        }
    }
}

// 1.2.1  LR Finder

// export
extension Learner where Opt.Scalar: BinaryFloatingPoint {
    public class LRFinder: Delegate {
        public typealias ScheduleFunc = (Float) -> Float

        // A learning rate schedule from step to float.
        private var scheduler: ScheduleFunc
        private var numIter: Int
        private var minLoss: Float? = nil
        
        public init(start: Float = 1e-5, end: Float = 10, numIter: Int = 100) {
            scheduler = makeAnnealer(start: start, end: end, schedule: expSchedule)
            self.numIter = numIter
        }
        
        override public func batchWillStart(learner: Learner) {
            learner.opt.learningRate = Opt.Scalar(scheduler(Float(learner.currentIter)/Float(numIter)))
        }
        
        override public func batchDidFinish(learner: Learner) throws {
            if minLoss == nil {minLoss = learner.currentLoss.scalar}
            else { 
                if learner.currentLoss.scalarized() < minLoss! { minLoss = learner.currentLoss.scalarized()}
                if learner.currentLoss.scalarized() > 4 * minLoss! { 
                    throw LearnerAction.stop(reason: "Loss diverged")
                }
                if learner.currentIter >= numIter { 
                    throw LearnerAction.stop(reason: "Finished the range.") 
                }
            }
        }
        
        override public func validationWillStart(learner: Learner<Label, Opt>) throws {
            //Skip validation during the LR range test
            throw LearnerAction.skipEpoch(reason: "No validation in the LR Finder.")
        }
    }
    
    public func makeLRFinder(start: Float = 1e-5, end: Float = 10, numIter: Int = 100) -> LRFinder {
        return LRFinder(start: start, end: end, numIter: numIter)
    }
}

// export
//TODO: when Recorder is a property of Learner don't return it.
extension Learner where Opt.Scalar: PythonConvertible & BinaryFloatingPoint {
    public func lrFind(start: Float = 1e-5, end: Float = 10, numIter: Int = 100) -> Recorder {
        let epochCount = data.train.count/numIter + 1
        let recorder = makeDefaultDelegates()
        delegates.append(makeLRFinder(start: start, end: end, numIter: numIter))
        try! self.fit(epochCount)
        return recorder
    }
}

