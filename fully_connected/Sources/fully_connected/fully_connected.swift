
import TensorFlow

// ## The forward and backward passes
// Typing `Tensor<Float>` all the time is tedious. 
// The S4TF team expects to make `Float` be the default so we can just say `Tensor`.  
// Until that happens though, we can define our own alias.
struct fully_connected {
    var text = "Hello, fully_connected!"
}

// export
public typealias TF=Tensor<Float>

// We will need to normalize our data.

// export
public func normalize(_ x:TF, mean:TF, std:TF) -> TF {
    return (x-mean)/std
}

//export
public func testNearZero(_ a: TF, tolerance: Float = 1e-3) {
    assert((abs(a) .< tolerance).all(), "Near zero: \(a)")
}

public func testSame(_ a: TF, _ b: TF) {
    // Check shapes match so broadcasting doesn't hide shape errors.
    assert(a.shape == b.shape)
    testNearZero(a-b)
}

// ## Foundations version
// ### Basic architecture

public func lin(_ x: TF, _ w: TF, _ b: TF) -> TF { return x•w+b }

public func myRelu(_ x:TF) -> TF { return max(x, 0) }

// ### Loss function
// We begin with the mean squared error to have easier gradient computations.

// export
public func mse(_ out: TF, _ targ: TF) -> TF {
    return (out.squeezingShape(at: -1) - targ).squared().mean()
}

// ## Gradients and backward pass
// Here we show how to calculate gradients for a simple model the hard way, manually.

// To store the gradients a bit like in PyTorch we introduce 
// a `TFGrad` class that has two attributes: the original tensor 
// and the gradient. We choose a class to easily replicate 
// the Python notebook: classes are reference types 

// (which means they are mutable) while structures are value types.
// In fact, since this is the first time we're discovering Swift classes, 
// let's jump into a 
// [sidebar discussion about Value Semantics vs Reference Semantics](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.g5669969ead_0_145) since it is a pretty fundamental part of the programming model and a huge step forward that Swift takes.
// When we get back, we'll keep charging on, even though this is very non-idiomatic Swift code!

/// WARNING: This is designed to be similar to the PyTorch 02_fully_connected lesson,
/// this isn't idiomatic Swift code.
public class TFGrad {
    public var inner, grad:  TF
    
    public init(_ x: TF) {
        inner = x
        grad = TF(zeros: x.shape)
    } 
}

// Redefine our functions on TFGrad.
public func lin(_ x: TFGrad, _ w: TFGrad, _ b: TFGrad) -> TFGrad {
    return TFGrad(x.inner • w.inner + b.inner)
}
public func myRelu(_ x: TFGrad) -> TFGrad {
    return TFGrad(max(x.inner, 0))
}
public func mse(_ inp: TFGrad, _ targ: TF) -> TF {
    //grad of loss with respect to output of previous layer
    return (inp.inner.squeezingShape(at: -1) - targ).squared().mean()
}
// Define our gradient functions.
public func mseGrad(_ inp: TFGrad, _ targ: TF) {
    //grad of loss with respect to output of previous layer
    inp.grad = 2.0 * (inp.inner.squeezingShape(at: -1) - targ)
        .expandingShape(at: -1) / Float(inp.inner.shape[0])
}

public func reluGrad(_ inp: TFGrad, _ out: TFGrad) {
    //grad of relu with respect to input activations
    inp.grad = out.grad.replacing(with: TF(zeros: inp.inner.shape), where: (inp.inner .< 0))
}

// This is our python version (we've renamed the python `g` to `grad` for consistency):
// ```python
// def lin_grad(inp, out, w, b):
//     inp.grad = out.grad @ w.t()
//     w.grad = (inp.unsqueeze(-1) * out.grad.unsqueeze(1)).sum(0)
//     b.grad = out.grad.sum(0)
// ```
public func linGrad(_ inp:TFGrad, _ out:TFGrad, _ w:TFGrad, _ b:TFGrad){
    // grad of linear layer with respect to input activations, weights and bias
    inp.grad = out.grad • w.inner.transposed()
    w.grad = inp.inner.transposed() • out.grad
    b.grad = out.grad.sum(squeezingAxes: 0)
}

/** 
///////////////////////////////////////////////////////////////////////////////
// # Automatic Differentiation in Swift

// There are a few challenges with the code above:

//  * It doesn't follow the principle of value semantics, because TensorGrad is a class. 
//  * Mutating a tensor would produce the incorrect results.
//  * It doesn't compose very well - we need to keep track of 
//  * values passed in the forward pass and also pass them in the backward pass.
//  * It is fully dynamic, keeping track of gradients at runtime. 
//  * This interferes with the compiler's ability to perform fusion 
//  * and other advanced optimizations.
 
// We want something that is simple, consistent and easy to use, like this:

// ## Autodiff the Functional Way

// Swift for TensorFlow's autodiff is built on value semantics and functional programming ideas.

// Each differentiable function gets an associated "chainer" (described below) 
// that defines its gradient.  When you write a function that, 
// like `model`, calls a bunch of these in sequence, 
// the compiler calls the function and it collects its pullback, 
// then stitches together the pullbacks using the chain rule from Calculus.

// Let's remember the chain rule - it is written:

// $$\frac{d}{dx}\left[f\left(g(x)\right)\right] = f'\left(g(x)\right)g'(x)$$

// Notice how the chain rule requires mixing together expressions from 
// both the forward pass (`g()`) and the backward pass (`f'()` and `g'()`) of a computation 
// to get the derivative.  While it is possible to calculate all the forward versions 
// of a computation, then recompute everything needed again on the backward pass, 
// this would be incredibly inefficient - it makes more sense to save intermediate 
// values from the forward pass and reuse them on the backward pass.

// The Swift language provides the atoms we need to express this: 
// we can represent math with function calls, and the pullback can be represented with a closure.  
// This works out well because closures provide a natural way to capture interesting 
// values from the forward pass.
To explore this, let's look at a really simple example of this, the inner computation of MSE.  The full body of MSE looks like this:

```swift
func mse(_ inp: TF, _ targ: TF) -> TF {
    //grad of loss with respect to output of previous layer
    return (inp.squeezingShape(at: -1) - targ).squared().mean()
}
```
 ## Basic expression in MSE
For the purposes of our example, we're going to keep it super super simple and 
just focus on the `x.squared().mean()` part of the computation, 
which we'll write as `mseInner(x) = mean(square(x))` to align better 
with function composition notation.  We want a way to visualize what functions get called, 
so let's define a little helper that prints the name of its caller whenever it is called. 
To do this we use a [litteral expression](https://docs.swift.org/swift-book/ReferenceManual/Expressions.html#ID390)
 `#function` that contains the name of the function we are in.
*/

// This function prints out the calling function's name.  This 
// is useful to see what is going on in your program..
public func trace(function: String = #function) {
  print(function)
}

// Ok, given that, we start by writing the implementation and gradients of these functions, 
// and we put print statements in them so we can tell when they are called. This looks like:
public func square(_ x: TF) -> TF {
    trace() 
    return x * x
}
public func 𝛁square(_ x: TF) -> TF {
    trace()
    return 2 * x
}

public func mean(_ x: TF) -> TF {
    trace()
    return x.mean()  // this is a reduction.  (can someone write this out longhand?)
}
public func 𝛁mean(_ x: TF) -> TF {
    trace()
    return TF(ones: x.shape) / Float(x.shape[0])
}

/** 
// Given these definitions we can now compute the forward and derivative 
// of the `mseInner` function that composes `square` and `mean`, using the chain rule:

// $$\frac{d}{dx}\left[f\left(g(x)\right)\right] = f'\left(g(x)\right)g'(x)$$

// where `f` is `mean` and `g` is `square`.  This gives us:
*/
public func mseInner(_ x: TF) -> TF {
    return mean(square(x))
}

public func 𝛁mseInner(_ x: TF) -> TF {
    return 𝛁mean(square(x)) * 𝛁square(x)
}

/**
 * This is all simple, but we have a small problem if (in the common case for deep nets) 
 * we want to calculate both the forward and the gradient computation 
 * at the same time: we end up redundantly computing square(x) 
 * in both the forward and backward paths!
 */
public func mseInnerAndGrad(_ x: TF) -> (TF, TF) {
  return (mseInner(x), 𝛁mseInner(x))    
}

/**
 * ## Reducing recomputation with Chainers and the ValueAndChainer pattern
We can fix this by refactoring our code.  
We want to preserve the linear structure of `mseInner` that calls `square` 
and then `mean`, but we want to make it so the ultimate *user* 
of the computation can choose whether they want the gradient computation (or not) and if so,
 we want to minimize computation.  To do this, we have to slightly 
 generalize our derivative functions.  While it is true that the derivative 
 of $square(x)$ is `2*x`, this is only true for a given point `x`.

If we generalize the derivative of `square` to work with an arbitrary **function**, 
instead of point, then we need to remember that $\frac{d}{dx}x^2 = 2x\frac{d}{dx}$, 
and therefore the derivative for `square` needs to get $\frac{d}{dx}$ passed in 
from its nested function.  

This form of gradient is known by the academic term "Vector Jacobian Product" (vjp) 
or the technical term "pullback", but we will refer to it as a 𝛁Chain because 
it implements the gradient chain rule for the operation.  We can write it like this:
 */

// The chainer for the gradient of square(x).
public func square𝛁Chain(x: TF, ddx: TF) -> TF {
  trace()
  return ddx * 2*x
}

// The chainer for the gradient of mean(x).
public func mean𝛁Chain(x: TF, ddx: TF) -> TF {
  trace()
  return ddx * TF(ones: x.shape) / Float(x.shape[0])
}
/**Given this very general way of describing gradients,
 *  we now want to pull them together in a single bundle that we can keep track of: 
 * we do this by changing each atom of computation to return both 
 * a normal value with the 𝛁Chain closure that produces a piece of 
 * the gradient given the chained input.

We refer to this as a "Value With 𝛁Chain" function (since that is what it is) 
and abreviate this mouthful to "VWC". This is also an excuse to use labels in tuples,
 which are a Swift feature that is very useful for return values like this.

They look like this:
 * 
 */

// Returns x*x and the chain for the gradient of x*x.
public func squareVWC(_ x: TF) -> (value: TF, chain: (TF) -> TF) {
  trace()
  return (value: x*x,
          chain: { ddx in square𝛁Chain(x: x, ddx: ddx) })    
}

// Returns the mean of x and the chain for the mean.
public func meanVWC(_ x: TF) -> (value: TF, chain: (TF) -> TF) {
  trace()
  return (value: x.mean(),
          chain: { ddx in mean𝛁Chain(x: x, ddx: ddx) })
}
/**
 * Given this, we can now implement mseInner in the same way. 
 * Notice how our use of named tuple results make the code nice and tidy:
 */
// We implement mean(square(x)) by calling each of the VWCs in turn.
public func mseInnerVWC(_ x: TF) -> (value: TF, chain: (TF) -> TF) {
  // square and mean are tuples that carry the value/chain for each step.
  let square = squareVWC(x)
  let mean   = meanVWC(square.value)

  // The result is the combination of the results and the pullbacks.
  return (mean.value,
          // The mseInner pullback calls the functions in reverse order.
          { v in square.chain(mean.chain(v)) })
}

/**
 * # Implementing Relu, MSE, and Lin with the Value With 𝛁Chain pattern
 * Lets come back to our earlier examples and define pullbacks for our primary functions 
 * in the simple model function example.
 */
public func reluVWC(_ x: TF) -> (value: TF, chain: (TF) -> TF) {
    return (value: max(x, 0),
            // Pullback for max(x, 0)
            chain: { 𝛁out -> TF in
              𝛁out.replacing(with: TF(zeros: x.shape), where: x .< 0)
            })
}
/**
 * ```swift
func lin(_ x: TFGrad, _ w: TFGrad, _ b: TFGrad) -> TFGrad {
    return TFGrad(x.inner • w.inner + b.inner)
}
func linGrad(_ inp:TFGrad, _ out:TFGrad, _ w:TFGrad, _ b:TFGrad){
    inp.grad = out.grad • w.inner.transposed()
    w.grad = inp.inner.transposed() • out.grad
    b.grad = out.grad.sum(squeezingAxes: 0)
}
```
 */
public func linVWC(_ inp: TF, _ w: TF, _ b: TF) -> (value: TF, chain: (TF) -> (TF, TF, TF)) {
    return (value: inp • w + b,
            // Pullback for inp • w + b.  Three results because 'lin' has three args.
            chain: { 𝛁out in
              (𝛁out • w.transposed(), 
               inp.transposed() • 𝛁out,
               𝛁out.unbroadcasted(to: b.shape))
    })
}
public func mseVWC(_ inp: TF, _ targ: TF) -> (value: TF, chain: (TF) -> (TF)) {
    let tmp = inp.squeezingShape(at: -1) - targ
    
    // We already wrote a VWC for x.square().mean(), so we can reuse it.
    let mseInner = mseInnerVWC(tmp)
    
    // Return the result, and a pullback that expands back out to
    // the input shape.
    return (mseInner.value, 
            { v in mseInner.chain(v).expandingShape(at: -1) })
}
/**
 * Ok, this is pretty nice - we get composition, we get value semantics, 
 * and everything just stacks up nicely.  We have a problem them, 
 * which is that this is a real pain to write and it is very easy to make simple mistakes.  
 * This is also very mechanical - and thus boring.

This is where Swift's autodiff system comes to the rescue!

# Automatically generating 𝛁Chains and VWCs

When you define a function with `@differentiable` you're saying 
that it must be differentiable by the compiler by composing 
the VWCs of other functions just like we did above manually.  
as it turns out, all of the methods on `Tensor` are marked up 
with `@differentiable` attributes until you get down to the atoms of the raw ops.  
For example, this is how the `Tensor.squared` method is 
[defined in Math.swift in the TensorFlow module](https://github.com/tensorflow/swift-apis/blob/12d93f22597d4502dfdbe59cb0d243e0828cdfb6/Sources/TensorFlow/Operators/Math.swift#L944):

```swift
// slightly simplified for clarity
public extension Tensor {
  @differentiable
  func squared() -> Tensor {
    return _Raw.square(self)
  }
}
```
   
The Value with 𝛁Chain function is 
[defined in Math.swift](https://github.com/tensorflow/swift-apis/blob/12d93f22597d4502dfdbe59cb0d243e0828cdfb6/Sources/TensorFlow/Operators/Math.swift#L952):

```swift
public extension Tensor {
  @derivative(of: squared)
  func _vjpSquared() -> (Tensor, (Tensor) -> Tensor) {
    return (squared(), { 2 * self * $0 })
  }
}
```

This tells the compiler that `squared()` has a manually 
written VJP that is implemented as we already saw.  Now, anything 
that calls `squared()` can have its own VJP synthesized out of it. 
 For example we can write our `mseInner` function the trivial way, 
*/
// The @differentiable attribute is normally optional in a S4TF standalone environment, 
// but is currently required in Jupyter notebooks. 
// The S4TF team is planning to relax this limitation when time permits.
@differentiable
public func mseInnerForAD(_ x: TF) -> TF {
    return x.squared().mean()
}

// because the compiler knows the VWCs for the squared and mean function, 
// it can synthesize them as we need them. Most often though, 
// you don't use the 𝛁Chain function directly. 
// You can instead ask for both the value and the gradient of a function at a specific point, 
// which is the most typical thing you'd use:

// # Bundling up a model into an aggregate value

// When we work with models and individual layers, 
// we often want to bundle up a bunch of differentiable variables into one value, 
// so we don't have to pass a ton of arguments around.  
// When we get to building our whole model, 
// it is mathematically just a struct that contains 
// a bunch of differentiable values embedded into it.  
// It is more convenient to think of a model as a function that 
// takes one value and returns one value rather than something 
// that can take an unbounded number of inputs: our simple model has 4 parameters, 
// and two normal inputs!
@differentiable
public func forward(_ inp: TF, _ targ: TF, w1: TF, b1: TF, w2: TF, b2: TF) -> TF {
    // FIXME: use lin
    let l1 = matmul(inp, w1) + b1
    let l2 = relu(l1)
    let l3 = matmul(l2, w2) + b2
    return (l3.squeezingShape(at: -1) - targ).squared().mean()
}


// Let's try refactoring our single linear model to use a `struct` to simplify this.  
// We start by defining a structure to contain all the fields we need.  
// We mark the structure as `: Differentiable` so the compiler knows 
// we want it to be differentiable (not discrete):
public struct MyModel: Differentiable {
    public var w1, b1, w2, b2: TF
    
    public init(_ ww1:TF,_ bb1:TF,_ ww2:TF,_ bb2:TF){
        w1 = ww1
        b1 = bb1
        w2 = ww2
        b2 = bb2
    }
}
// We can now define our forward function as a method on this model:
public extension MyModel {
    @differentiable
     func forward(_ input: TF, _ target: TF) -> TF {
        // FIXME: use lin
        let l1 = matmul(input, w1) + b1
        let l2 = relu(l1)
        let l3 = matmul(l2, w2) + b2
        // use mse
        return (l3.squeezingShape(at: -1) - target).squared().mean()
    }
}
// Given this, we can now get the gradient of our entire loss w.r.t 
// to the input and the expected labels:

/**
 * # More about AutoDiff

There are lots of cool things you can do with Swift autodiff.  
One of the great things about understanding how the system fits together is that you do 
a lot of interesting things by customizing gradients with S4TF.  
This can be useful for lots of reasons, for example:

 * you want a faster approximation of an expensive gradient
 * you want to improve the numerical instability of a gradient
 * you want to pursue exotic techniques like learned gradients
 * you want to work around a limitation of the current implementation
 
In fact, we've had to do that in `11_imagenette` where we've built a `SwitchableLayer` 
with a custom gradient.  Let's go take a look.

To find out more, check out this nice tutorial in Colab on 
[custom autodiff](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/custom_differentiation.ipynb)
 */

// import NotebookExport
// let exporter = NotebookExport(Path.cwd/"02_fully_connected.ipynb")
// print(exporter.export(usingPrefix: "FastaiNotebook_"))