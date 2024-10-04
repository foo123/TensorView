# TensorView

View one-dimensional array data, typed array data and/or multi-dimensional array data as multidimensional tensors of various shapes efficiently.

version: **1.0.0** (8.9 kB minified)

`TensorView` is both memory-efficient and speed-efficient since it only creates ways to view array data as multidimensional tensors **without** actually creating new arrays. One can nevertheless explicitly store a TensorView instance as a single-dimensional or multi-dimensional array using `view.toArray()` or `view.toNDArray()` methods.

**Example** (see `/test/demo.js`)

```javascript
const TensorView = require('../src/TensorView.js');

const array = [1,2,3,4,5,6]; // single-dimensional data
const ndarray = [[1,2,3],[4,5,6]]; // multi-dimensional data

const s = TensorView(array, {shape:[2,3]}); // create a view with shape
const sT = s.transpose(); // get transposed view

const m = TensorView(ndarray, {ndarray:[2,3],shape:[3,2]}); // create view of ndarray with different shape
const m2 = m.reshape([2,3]).slice([null,{start:1,stop:2,step:1}]); // reshape and get a slice

console.log(s.toNDArray());
console.log(s.toArray());
console.log(sT.toNDArray());
console.log(sT.toArray());
console.log(s.data() === sT.data()) // uses same data
console.log(m.toNDArray());
console.log(m.toArray());
console.log(m2.toNDArray());
console.log(m2.toArray());
console.log(m.data() === m2.data()) // uses same data

// iterator protocol
for (let [data_i, i] of s) console.log([data_i, i]); // i is multi-dimensional index in general

// same as
s.forEach((data_i, i) => console.log([data_i, i])); // i is multi-dimensional index in general
```

**Output**

```text
[ [ 1, 2, 3 ], [ 4, 5, 6 ] ]
[ 1, 2, 3, 4, 5, 6 ]
[ [ 1, 4 ], [ 2, 5 ], [ 3, 6 ] ]
[ 1, 4, 2, 5, 3, 6 ]
true
[ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]
[ 1, 2, 3, 4, 5, 6 ]
[ [ 2, 3 ], [ 5, 6 ] ]
[ 2, 3, 5, 6 ]
true
[ 1, [ 0, 0 ] ]
[ 2, [ 0, 1 ] ]
[ 3, [ 0, 2 ] ]
[ 4, [ 1, 0 ] ]
[ 5, [ 1, 1 ] ]
[ 6, [ 1, 2 ] ]
[ 1, [ 0, 0 ] ]
[ 2, [ 0, 1 ] ]
[ 3, [ 0, 2 ] ]
[ 4, [ 1, 0 ] ]
[ 5, [ 1, 1 ] ]
[ 6, [ 1, 2 ] ]
```

**Methods:**

```javascript
// data=single value or single-dimensional array or multi-dimensional array
// options={shape?:Array, slice?:Array, ndarray?:Array}
// shape array defines shape of view
// slice defines slices along various axes to be part of the view (default is whole shape)
// ndarray defines the shape of multi-dimensional array data if such array is passed as data
const view  = TensorView(data, options);

const data = view.data(); // underlying data of view
const array = view.toArray(ArrayClass=Array); // create single-dimensional array or typed array from view
const ndarray = view.toNDArray(); // create multi-dimensional array from view having the same shape
const string = view.toString(); // render view to string
const dim = view.dimension(); // dimension of view, eg 1 for 1d, 2 for 2d, 3 for 3d, ..
const shape = view.shape(); // shape of view along all dimensions
const shapeForAxis = view.shape(axis); // shape of view along `axis` dimension
const size = view.size(); // actual size of view along all dimensions (if slicing is active size is different than shape)
const sizeForAxis = view.size(axis); // size of view along `axis` dimension
const length = view.length(); // actual length of view (eg if saved as array)
const slices = view.slicing(); // the slicing applied along all dimensions
const sliceForAxis = view.slicing(axis); // the slicing applied along `axis` dimension

const transpose = view.transpose(); // transpose view
const reshaped = view.reshape(newShape); // re-shaped view
const slice = view.slice([{start:a,stop:b,step:c}|null,..]); // get a sliced view (stop is included), same as view[a:b+1:c,:,..]
const concatenated = view.concat([view2, view3, ..], axis=0); // concatenate multiple similar views along some `axis` axis

const value = view.get(indices); // get value based on indices of same dimension as view shape
view.set(indices, value); // set value at indices
// NOTE: underlying data will change in all views which use this data and all views which depend on views which use this data

view.forEach(function(data_i, i, data, view) {/*..*/}); // forEach method
for (let [data_i, i] of view) {/*..*/} // similar as iterator protocol

view.op(op, otherView=null); // apply lazy, when requested, pointwise operation op(view, otherView) or op(view)

// creating an actual copy and not share data is easy to do in various ways, eg:
const copied = TensorView(view.toArray(), {shape:view.size()}); // any active slicing and/or operation will be applied on view output

view.dispose(); // dispose view if no longer needed
```

**see also:**

* [Abacus](https://github.com/foo123/Abacus) advanced Combinatorics and Algebraic Number Theory Symbolic Computation library for JavaScript, Python
* [TensorView](https://github.com/foo123/TensorView) view array data as multidimensional tensors of various shapes efficiently
* [Geometrize](https://github.com/foo123/Geometrize) Computational Geometry and Rendering Library for JavaScript
* [Plot.js](https://github.com/foo123/Plot.js) simple and small library which can plot graphs of functions and various simple charts and can render to Canvas, SVG and plain HTML
* [CanvasLite](https://github.com/foo123/CanvasLite) an html canvas implementation in pure JavaScript
* [Rasterizer](https://github.com/foo123/Rasterizer) stroke and fill lines, rectangles, curves and paths, without canvas
* [Gradient](https://github.com/foo123/Gradient) create linear, radial, conic and elliptic gradients and image patterns without canvas
* [css-color](https://github.com/foo123/css-color) simple class to parse and manipulate colors in various formats
* [MOD3](https://github.com/foo123/MOD3) 3D Modifier Library in JavaScript
* [HAAR.js](https://github.com/foo123/HAAR.js) image feature detection based on Haar Cascades in JavaScript (Viola-Jones-Lienhart et al Algorithm)
* [HAARPHP](https://github.com/foo123/HAARPHP) image feature detection based on Haar Cascades in PHP (Viola-Jones-Lienhart et al Algorithm)
* [FILTER.js](https://github.com/foo123/FILTER.js) video and image processing and computer vision Library in pure JavaScript (browser and node)
* [Xpresion](https://github.com/foo123/Xpresion) a simple and flexible eXpression parser engine (with custom functions and variables support), based on [GrammarTemplate](https://github.com/foo123/GrammarTemplate), for PHP, JavaScript, Python
* [Regex Analyzer/Composer](https://github.com/foo123/RegexAnalyzer) Regular Expression Analyzer and Composer for PHP, JavaScript, Python
* [GrammarTemplate](https://github.com/foo123/GrammarTemplate) grammar-based templating for PHP, JavaScript, Python
* [codemirror-grammar](https://github.com/foo123/codemirror-grammar) transform a formal grammar in JSON format into a syntax-highlight parser for CodeMirror editor
* [ace-grammar](https://github.com/foo123/ace-grammar) transform a formal grammar in JSON format into a syntax-highlight parser for ACE editor
* [prism-grammar](https://github.com/foo123/prism-grammar) transform a formal grammar in JSON format into a syntax-highlighter for Prism code highlighter
* [highlightjs-grammar](https://github.com/foo123/highlightjs-grammar) transform a formal grammar in JSON format into a syntax-highlight mode for Highlight.js code highlighter
* [syntaxhighlighter-grammar](https://github.com/foo123/syntaxhighlighter-grammar) transform a formal grammar in JSON format to a highlight brush for SyntaxHighlighter code highlighter
* [SortingAlgorithms](https://github.com/foo123/SortingAlgorithms) implementations of Sorting Algorithms in JavaScript
* [PatternMatchingAlgorithms](https://github.com/foo123/PatternMatchingAlgorithms) implementations of Pattern Matching Algorithms in JavaScript
