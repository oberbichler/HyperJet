# HyperJet
> Automatic differentiation with hyper-dual numbers

## What it is

`HyperJet` is a small but powerful library to enable automatic computation of derivatives. It implements a basic _forward mode_ automatic differentiation algorithm to compute the _first and second derivative_ of an expression with respect to multiple input variables.

The library was initially developed to simplify the _development of isogeometric finite elements_. It can be easily integrated into existing _Python_ and _C++_ code.

## Similar Tools

There are many tools to compute derivatives automatically. Each is suitable for specific use cases. If HyperJet is not the right thing for you, you might should check out [this page](http://www.autodiff.org/?module=Tools).

## Reference

If you use HyperJet, please refer to the official GitHub repository:

```
@misc{HyperJet,
  author = "Thomas Oberbichler",
  title = "HyperJet",
  howpublished = "\url{http://github.com/oberbichler/HyperJet}",
}
```