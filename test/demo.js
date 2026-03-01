"use strict";

const TensorView = require('../src/TensorView.js');
const echo = console.log;

const array = [1,2,3,4,5,6,7,8,9,0]; // single-dimensional data
const ndarray = [[1,2,3,4,5],[6,7,8,9,0]]; // multi-dimensional data

// transposes
const s = TensorView(array, {shape:[2,5]}); // create a view with shape
const sT = s.transpose(); // get transposed view
echo('transposes');
echo(s.toNDArray());
echo(s.toArray());
echo(sT.toNDArray());
echo(sT.toArray());
echo(s.data === sT.data) // uses same data

echo('---');

// iterator protocol
for (let [data_i, i] of s) echo([data_i, i.slice()]); // i is multi-dimensional index in general
echo();
// same as
s.forEach((data_i, i) => echo([data_i, i.slice()])); // i is multi-dimensional index in general
echo();
for (let [data_i, i] of sT) echo([data_i, i.slice()]); // i is multi-dimensional index in general
echo();
// same as
sT.forEach((data_i, i) => echo([data_i, i.slice()])); // i is multi-dimensional index in general

// slices and nested slices
const s1 = TensorView(array, {shape:[2,5]});
const s2 = s1.slice(':','2:4');
const s3 = s2.slice('1', '-1:-1:0').squeeze();
const s4 = s2.slice('1', '-1,0').squeeze();

echo('---');

echo('slices');
echo(s1.toNDArray());
echo(s2.toNDArray());
echo(s3.toNDArray());
echo(s4.toNDArray());
echo(s1.data === s2.data, s1.data === s3.data, s1.data === s4.data); // uses same data

// concatenations
const c1 = TensorView(array, {shape:[2,5]});
const c2 = c1.concat(c1, 0);
const c3 = c2.slice(':', '2:-1:1').squeeze(); // get a slice and squeeze
const A = TensorView([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
const B = TensorView([[0, 5, 4], [2, 7, 6], [9, 2, 1]]);
const C = A.concat(B, "newaxis");

echo('---');

echo('concatenations');
echo(c1.toNDArray());
echo(c2.toNDArray());
echo(c3.toNDArray());
echo('---');
echo(A.toString());
echo(B.toString());
echo(C.toString());

const P1 = C.permute([1, 0, 2]); // interchange rows/columns
echo('---');

echo('permutations');
echo(C.toString());
echo(P1.toString());

const m = TensorView(ndarray, {shape:[5,2]}); // create view of ndarray with different shape
const m2 = m.reshape([2,5]); // reshape
const m3 = m2.slice(':', '1:2'); // get a slice

echo('---');

echo('reshape');
echo(m.toNDArray());
echo(m.toArray());
echo(m2.toNDArray());
echo(m2.toArray());
echo(m3.toNDArray());
echo(m3.toArray());
echo(m.data === m2.data, m.data === m3.data) // uses same data

const a = TensorView([1,2,3,4,5,6], {shape:[2,3]});
const c = a.transpose();
const b = a.reorder("column-major", "row-major");
echo('---');

echo('reorder');
echo(a.toNDArray());
echo(b.toNDArray());
echo(c.toNDArray());
echo(a.toArray());
echo(b.toArray());
echo(c.toArray());
