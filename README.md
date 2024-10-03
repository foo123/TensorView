# TensorView

View one-dimensional array data, typed array data and/or multi-dimensional array data as multidimensional tensors of various shapes efficiently.

version: **1.0.0**

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
```