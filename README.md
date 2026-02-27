# TensorView

View one-dimensional array data, typed array data and/or multi-dimensional array data as multidimensional tensors of various shapes efficiently.

![TensorView](/tensorview.jpg)

version: **2.0.0** (9 kB minified)

`TensorView` is both memory-efficient and speed-efficient since it only creates ways to view array data as multidimensional tensors **without** actually creating new arrays. One can nevertheless explicitly store a TensorView instance as a single-dimensional or multi-dimensional array using `view.toArray()` or `view.toNDArray()` methods.

**Example** (see `/test/demo.js`)

```javascript
const TensorView = require('../src/TensorView.js');

const array = [1,2,3,4,5,6,7,8,9,0]; // single-dimensional data
const ndarray = [[1,2,3,4,5],[6,7,8,9,0]]; // multi-dimensional data

const s = TensorView(array, {shape:[2,5]}); // create a view with shape
const sT = s.transpose(); // get transposed view

console.log(s.toNDArray());
console.log(s.toArray());
console.log(sT.toNDArray());
console.log(sT.toArray());
console.log(s.data === sT.data) // uses same data

const m = TensorView(ndarray, {shape:[5,2]}); // create view of ndarray with different shape
const m2 = m.reshape([2,5]); // reshape
const m3 = m2.slice(':', '1:2'); // get a slice

console.log(m.toNDArray());
console.log(m.toArray());
console.log(m2.toNDArray());
console.log(m2.toArray());
console.log(m3.toNDArray());
console.log(m3.toArray());
console.log(m.data === m2.data, m.data === m3.data) // uses same data

// iterator protocol
for (let [data_i, i] of s) console.log([data_i, i.slice()]); // i is multi-dimensional index in general

// same as
s.forEach((data_i, i) => console.log([data_i, i.slice()])); // i is multi-dimensional index in general
```

**Output**

```text
[ [ 1, 2, 3, 4, 5 ], [ 6, 7, 8, 9, 0 ] ]
[
  1, 2, 3, 4, 5,
  6, 7, 8, 9, 0
]
[ [ 1, 6 ], [ 2, 7 ], [ 3, 8 ], [ 4, 9 ], [ 5, 0 ] ]
[
  1, 6, 2, 7, 3,
  8, 4, 9, 5, 0
]
true

[ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ], [ 7, 8 ], [ 9, 0 ] ]
[
  1, 2, 3, 4, 5,
  6, 7, 8, 9, 0
]
[ [ 1, 2, 3, 4, 5 ], [ 6, 7, 8, 9, 0 ] ]
[
  1, 2, 3, 4, 5,
  6, 7, 8, 9, 0
]
[ [ 2, 3 ], [ 7, 8 ] ]
[ 2, 3, 7, 8 ]
true true

[ 1, [ 0, 0 ] ]
[ 2, [ 0, 1 ] ]
[ 3, [ 0, 2 ] ]
[ 4, [ 0, 3 ] ]
[ 5, [ 0, 4 ] ]
[ 6, [ 1, 0 ] ]
[ 7, [ 1, 1 ] ]
[ 8, [ 1, 2 ] ]
[ 9, [ 1, 3 ] ]
[ 0, [ 1, 4 ] ]
[ 1, [ 0, 0 ] ]
[ 2, [ 0, 1 ] ]
[ 3, [ 0, 2 ] ]
[ 4, [ 0, 3 ] ]
[ 5, [ 0, 4 ] ]
[ 6, [ 1, 0 ] ]
[ 7, [ 1, 1 ] ]
[ 8, [ 1, 2 ] ]
[ 9, [ 1, 3 ] ]
[ 0, [ 1, 4 ] ]

```

**Methods:**

```javascript
// data=single value or single-dimensional array or typed array or multi-dimensional array
// options={shape?:Array}
// shape array defines shape of view
const view  = TensorView(data, options);

const data = view.data; // underlying data of view
const dim = view.dimension; // dimension of view, eg 1 for 1d, 2 for 2d, 3 for 3d, ..
const length = view.length; // actual length of view (eg if saved as array)

const shape = view.shape(); // shape of view along all dimensions
const shapeForAxis = view.shape(axis); // shape of view along `axis` dimension

const array = view.toArray(ArrayClass=Array, order="row-major"); // create single-dimensional array or typed array from view
const ndarray = view.toNDArray(order="row-major"); // create multi-dimensional array from view having the same shape
const string = view.toString(); // render view to string

const transpose = view.transpose(); // transposed view
const reshaped = view.reshape(newShape); // view with different shape
const slice = view.slice(":", "a:b", "a:s:b", ..); // sliced view from a to b (included) with step s, ..
const concatenated = view.concat([view2, view3, ..], axis=0); // concatenate multiple similar views along some `axis` axis
const squeezed = view.squeeze(); // get view with any dimension along some axis of length 1 removed

const value = view.get(indices); // get value based on indices of same dimension as view shape
view.set(indices, value); // set value at indices
// NOTE: underlying data will change in all views which use this data and all views which depend on views which use this data

view.forEach(function(data_i, i, data, view) {/*..*/}, order="row-major"); // forEach method
for (let [data_i, i] of view) {/*..*/} // similar as iterator protocol

// creating an actual copy and not share data is easy to do in various ways, eg:
const copied = TensorView(view.toArray(), {shape: view.shape()});

view.dispose(); // dispose view if no longer needed
NOTE: will affect any other active views which depend on this view (eg concatenated views, sliced views, ..), so take note
```

**see also:**

* [Abacus](https://github.com/foo123/Abacus) Computer Algebra and Symbolic Computation System for Combinatorics and Algebraic Number Theory for JavaScript and Python
* [SciLite](https://github.com/foo123/SciLite) Scientific Computing Environment similar to Octave/Matlab in pure JavaScript
* [TensorView](https://github.com/foo123/TensorView) view array data as multidimensional tensors of various shapes efficiently
* [FILTER.js](https://github.com/foo123/FILTER.js) video and image processing and computer vision Library in pure JavaScript (browser and nodejs)
* [HAAR.js](https://github.com/foo123/HAAR.js) image feature detection based on Haar Cascades in JavaScript (Viola-Jones-Lienhart et al Algorithm)
* [HAARPHP](https://github.com/foo123/HAARPHP) image feature detection based on Haar Cascades in PHP (Viola-Jones-Lienhart et al Algorithm)
* [Fuzzion](https://github.com/foo123/Fuzzion) a library of fuzzy / approximate string metrics for PHP, JavaScript, Python
* [Matchy](https://github.com/foo123/Matchy) a library of string matching algorithms for PHP, JavaScript, Python
* [Regex Analyzer/Composer](https://github.com/foo123/RegexAnalyzer) Regular Expression Analyzer and Composer for PHP, JavaScript, Python
* [Xpresion](https://github.com/foo123/Xpresion) a simple and flexible eXpression parser engine (with custom functions and variables support), based on [GrammarTemplate](https://github.com/foo123/GrammarTemplate), for PHP, JavaScript, Python
* [GrammarTemplate](https://github.com/foo123/GrammarTemplate) grammar-based templating for PHP, JavaScript, Python
* [codemirror-grammar](https://github.com/foo123/codemirror-grammar) transform a formal grammar in JSON format into a syntax-highlight parser for CodeMirror editor
* [ace-grammar](https://github.com/foo123/ace-grammar) transform a formal grammar in JSON format into a syntax-highlight parser for ACE editor
* [prism-grammar](https://github.com/foo123/prism-grammar) transform a formal grammar in JSON format into a syntax-highlighter for Prism code highlighter
* [highlightjs-grammar](https://github.com/foo123/highlightjs-grammar) transform a formal grammar in JSON format into a syntax-highlight mode for Highlight.js code highlighter
* [syntaxhighlighter-grammar](https://github.com/foo123/syntaxhighlighter-grammar) transform a formal grammar in JSON format to a highlight brush for SyntaxHighlighter code highlighter
* [MOD3](https://github.com/foo123/MOD3) 3D Modifier Library in JavaScript
* [Geometrize](https://github.com/foo123/Geometrize) Computational Geometry and Rendering Library for JavaScript
* [Plot.js](https://github.com/foo123/Plot.js) simple and small library which can plot graphs of functions and various simple charts and can render to Canvas, SVG and plain HTML
* [CanvasLite](https://github.com/foo123/CanvasLite) an html canvas implementation in pure JavaScript
* [Rasterizer](https://github.com/foo123/Rasterizer) stroke and fill lines, rectangles, curves and paths, without canvas
* [Gradient](https://github.com/foo123/Gradient) create linear, radial, conic and elliptic gradients and image patterns without canvas
* [css-color](https://github.com/foo123/css-color) simple class to parse and manipulate colors in various formats
* [PatternMatchingAlgorithms](https://github.com/foo123/PatternMatchingAlgorithms) library of Pattern Matching Algorithms in JavaScript using [Matchy](https://github.com/foo123/Matchy)
* [SortingAlgorithms](https://github.com/foo123/SortingAlgorithms) library of Sorting Algorithms in JavaScript
