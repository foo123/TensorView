# TensorView

View one-dimensional array data, typed array data and/or multi-dimensional array data as multidimensional tensors of various shapes efficiently.

![TensorView](/tensorview.jpg)

version: **2.0.0** (12 kB minified)

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
for (let [item, index] of s) console.log([item, index.slice()]); // index is array of multidimensional indices

// same as
s.forEach((item, index) => console.log([item, index.slice()])); // index is array of multidimensional indices
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
// options={shape?:Array, stride?:Array, in_order?:Array, out_order?:Array}
// shape array defines desired shape of view
// stride array defines strides for each dimension of view (optional)
// in_order array defines order of dimension traversal when reading from data (optional)
// out_order array defines order of dimension traversal when writing to data (optional)
const view  = TensorView(data, options);

// underlying data of view
const data = view.data;
// dimension of view, eg 1 for 1d, 2 for 2d, 3 for 3d, ..
const dim = view.dimension;
// actual length of view (eg if saved as array)
const length = view.length;

// shape array of view along all dimensions
const shape = view.shape();
// shape of view along `axis` dimension
const shapeForAxis = view.shape(axis);

// stride array of view along all dimensions
const stride = view.stride();
// stride for `axis` dimension
const strideForAxis = view.stride(axis);

// create single-dimensional array or typed array from view
const array = view.toArray(ArrayClass=Array);

// create multi-dimensional array from view having the same shape
const ndarray = view.toNDArray();

// render view to string,
// maxSize defines max number of items to display along a dimension (default Infinity)
// stringify defines custom stringifier function (default toString)
const string = view.toString(maxSize=Infinity, stringify=String);

// transposed view
const transposed = view.transpose();

// view with different shape
const reshaped = view.reshape(new_shape);

// view with different in/out order
const reordered = view.reorder(new_in_order, new_out_order);

// view with permuted dimensions
const permuted = view.permute(permutation);

// sliced view with whole axis,
// or only a and b indices,
// or from indices a to b (included),
// or from indices a to b (included) with step s,
// etc..
const sliced = view.slice(":", "a,b,..", "a:b", "a:s:b" /*, ..*/);

// concatenate multiple views along on_axis axis or "newaxis"
const concatenated = view.concat([view2, view3 /*, ..*/], on_axis=0);

// get view with any dimension along some axis (after start_axis) of length 1 removed
const squeezed = view.squeeze(start_axis=0);

// get value based on multidimensional indices of same dimension as view shape
const value = view.get(indices);

// set value at multidimensional indices
view.set(indices, value);
// NOTE: underlying data will change in all views which use this data and all views which depend on views which use this data

// forEach method
view.forEach(function(item, index, data, view) {/*..*/});

// similar as iterator protocol
for (let [item, index] of view) {/*..*/}

// creating an actual copy and not share data is easy to do in various ways, eg:
const viewcopy = TensorView(view.toArray(), {shape: view.shape()});

// dispose view if no longer needed
view.dispose();
// NOTE: will affect any other active views which depend on this view (eg concatenated views, sliced views, ..), so take note
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
