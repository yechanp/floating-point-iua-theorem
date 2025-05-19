# Floating-Point Neural Networks Are Provably Robust Universal Approximators (CAV 2025)

[Geonho Hwang](https://sites.google.com/snu.ac.kr/geonhohwang),
[Wonyeol Lee](https://wonyeol.github.io/),
[Yeachan Park](https://yechanp.github.io/site/),
[Sejun Park](https://sites.google.com/site/sejunparksite/),
[Feras Saad](https://www.cs.cmu.edu/~fsaad/)


## Paper Abstract 

The classical universal approximation (UA) theorem for neural networks establishes mild conditions under which a feedforward neural network can approximate a continuous function f with arbitrary accuracy. A recent result establishes that neural networks also enjoy a more general interval universal approximation (IUA) theorem, in the sense that the abstract interpretation semantics of the network using the interval domain can approximate the direct image map of f (i.e., the result of applying f to a set of inputs) with arbitrary accuracy. These theorems, however, rest on the unrealistic assumption that the neural network computes over infinitely precise real numbers, whereas their software implementations in practice compute over finite-precision floating-point numbers. An open is whether the IUA theorem still holds in the floating-point setting.

This paper introduces the first IUA theorem for floating-point neural networks that proves their remarkable ability to perfectly capture the direct image map of any rounded target function f, showing no limits exist on their expressiveness. Our IUA theorem in the floating-point setting exhibits material differences from the real-valued setting, which reflects the fundamental distinctions between these two computational models. This theorem also implies surprising corollaries, which include (i) the existence of provably robust floating-point neural networks; and (ii) the computational completeness of the class of straight-line programs that use only floating-point additions and multiplications for the class of all floating-point programs that halt.


## Implementation Details

This repository provides the implementation and verification of a proposed neural network architecture designed to represent a given target function under interval analysis and floating-point arithmetic. The implementation assumes that both the domain and codomain are $[0,1]$, and it uses the float16 format. The network is constructed to represent the specified target function, and its correctness is verified through interval analysis by randomly selecting subintervals of $[0,1]$ for specified verification steps.

## How to Run

```
python interval.py -t sin -v 500 
```

-v : verification steps. 

-t : target function ( "square" for $f(x)=x^2$, "sin" for $f(x) = \sin( 10 \pi x ) /2$. )


